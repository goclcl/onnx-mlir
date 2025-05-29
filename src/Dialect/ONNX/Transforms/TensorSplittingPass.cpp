#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct TensorSplittingPass
    : public PassWrapper<TensorSplittingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorSplittingPass)

  StringRef getArgument() const override { return "onnx-tensor-splitting"; }

  StringRef getDescription() const override {
    return "Reduces memory bottlenecks by splitting tensors";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    bool changed = true; // 변경 여부를 추적

    // peak op 분석, 최적화를 반복함. 최적화 불가능하면 루프 종료
    while (changed) {
      changed = false;
      Liveness &liveness = getAnalysis<Liveness>();

      Operation *peakOp = nullptr;
      Operation *prePeakOp = nullptr;
      Operation *postPeakOp = nullptr;
      int64_t maxMemoryUsage = -1;

      // peak memory인 op를 구함
      funcOp.walk([&](Operation *op) {
        int64_t currentOpMemoryUsage = calculateMemoryUsageAtOp(op, liveness);
        if (currentOpMemoryUsage > maxMemoryUsage) {
          maxMemoryUsage = currentOpMemoryUsage;
          peakOp = op;
        }
        return WalkResult::advance();
      });

      Value peakOpOperand = peakOp->getOperand(0);
      prePeakOp = peakOpOperand.getDefiningOp(); // pwConv일때 최적화 가능
      postPeakOp = *peakOp->getResult(0).getUsers().begin();

      if (!peakOp) {
        llvm::outs()
            << "[TensorSplittingPass] No peak op found or function is empty.\n";
        break;
      }

      llvm::outs() << "[TensorSplittingPass] Pre peak op: " << *prePeakOp
                   << "\n";
      llvm::outs() << "                      Peak op: " << *peakOp << "\n";
      llvm::outs() << "                      Post peak op: " << *postPeakOp
                   << "\n";

      if (!isOptimizable(prePeakOp)) {
        llvm::outs() << "[TensorSplittingPass] Peak op is not optimizable.\n";
        break;
      }

      OpBuilder builder(prePeakOp->getContext());

      if (performTensorSplitting(prePeakOp, builder)) {
        static int64_t optCount = 1;
        changed = true; // IR이 변경되었음을 표시
        llvm::outs() << "[TensorSplittingPass] Optimization applied "
                     << optCount << " time" << (optCount == 1 ? "" : "s")
                     << ".\n\n";
        optCount++;
        // IR이 변경되었으므로 Liveness 분석을 다시 실행해야 함.
        // 루프가 다시 시작될 때 getAnalysis<Liveness>()가 호출되어 갱신됨.
      } else {
        // performTensorSplitting에서 false를 반환하면 더 이상 진행할 수
        // 없으므로 break.
        break;
      }
    } // End while
  }

private:
  int64_t getTensorSize(TensorType tensorType) {
    if (!tensorType || !tensorType.hasRank()) {
      return 0;
    }
    int64_t totalSize = 1;
    bool hasDynamicDim = false;
    for (int64_t dim : tensorType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        hasDynamicDim = true;
        break;
      }
      if (dim == 0)
        return 0; // 차원 중 하나가 0이면 전체 크기는 0
      totalSize *= dim;
    }
    // 동적 차원이 있는 경우 크기를 정확히 알 수 없으므로 1로 처리
    return hasDynamicDim ? 1 : totalSize;
  }

  int64_t calculateMemoryUsageAtOp(Operation *op, Liveness &liveness) {
    const LivenessBlockInfo *blockLiveness =
        liveness.getLiveness(op->getBlock());
    if (!blockLiveness)
      return 0;

    int64_t memoryUsage = 0;
    SmallPtrSet<Value, 16> liveValuesSet;

    // op 실행시 살아있는는 값들
    for (Value val : blockLiveness->currentlyLiveValues(op)) {
      liveValuesSet.insert(val);
    }
    // op 자체의 결과도 추가
    // liveValueSet에 이미 존재하더라도 set이라 중복 허용 안됨
    for (Value result : op->getResults()) {
      liveValuesSet.insert(result);
    }

    for (Value val : liveValuesSet) {
      if (auto tensorType = val.getType().dyn_cast<TensorType>()) {
        memoryUsage += getTensorSize(tensorType);
      }
    }
    return memoryUsage;
  }

  // isOptimizable은 prePeakOp가 pointwise conv이고, 그 다음 depthwise conv
  // (clip 포함일 수도 있음)가 오며, 중간 텐서가 최종 텐서보다 큰 경우를 확인함.
  bool isOptimizable(Operation *prePeakOp) {
    if (!isPointwiseConv(prePeakOp)) {
      llvm::outs() << "[TensorSplittingPass] Memory peak is not at the "
                      "pointwise Conv.\n";
      return false;
    }

    auto pwConv = llvm::dyn_cast<ONNXConvOp>(prePeakOp);
    if (!pwConv)
      return false;

    Value intermediateTensor = pwConv.getResult();
    Operation *nextOpAfterPw = nullptr;

    // pwConv의 출력이 하나의 연산에서만 사용되는지 확인
    if (!intermediateTensor.hasOneUse()) {
      llvm::outs()
          << "[TensorSplittingPass] Intermediate tensor has multiple uses.\n";
      return false;
    }

    Operation *user = *intermediateTensor.getUsers().begin();

    if (auto clipOp = llvm::dyn_cast<ONNXClipOp>(user)) {
      // ClipOp의 출력이 하나의 dwConv로만 가는지 확인
      if (clipOp.getResult().hasOneUse()) {
        Operation *clipUser = *clipOp.getResult().getUsers().begin();
        if (isDepthwiseConv(clipUser)) {
          nextOpAfterPw = clipUser;
        }
      } else {
        llvm::outs()
            << "[TensorSplittingPass] Intermediate tensor has multiple uses.\n";
        return false;
      }
    } else if (isDepthwiseConv(user)) {
      nextOpAfterPw = user;
    }

    if (!nextOpAfterPw) { // dwConv를 찾지 못함
      llvm::outs() << "[TensorSplittingPass] No depthwise convolution after "
                      "pointwise convolution.\n";
      return false;
    }

    // isIntermediateLargerThanDepthwise 내부에서 pwConv와 nextOpAfterPw
    // (dwConv)를 사용
    bool larger = isIntermediateLargerThanDepthwise(
        pwConv, llvm::dyn_cast<ONNXConvOp>(nextOpAfterPw));
    if (larger)
      return true;
    else {
      llvm::outs() << "[TensorSplittingPass] Intermediate tensor is not larger "
                      "than depthwise "
                      "convolution's result tensor.\n";
      return false;
    }
  }

  bool isPointwiseConv(Operation *op) {
    auto convOp = llvm::dyn_cast_or_null<ONNXConvOp>(op);
    if (!convOp)
      return false;

    Value kernel = convOp.getOperand(1);
    auto kernelType = kernel.getType().dyn_cast<RankedTensorType>();
    if (!kernelType)
      return false;

    ArrayRef<int64_t> shape = kernelType.getShape();
    unsigned rank = shape.size();
    // ONNX Conv 가중치 형태: [M, C_in/group, kH, kW]
    // Pointwise는 kH = 1, kW = 1
    return rank >= 2 && shape[rank - 1] == 1 && shape[rank - 2] == 1;
  }

  bool isDepthwiseConv(Operation *op) {
    auto convOp = llvm::dyn_cast_or_null<ONNXConvOp>(op);
    if (!convOp)
      return false;

    IntegerAttr groupAttr = convOp.getGroupAttr();
    if (!groupAttr)
      return false;
    int64_t group = groupAttr.getValue().getSExtValue();
    if (group <= 1)
      return false;

    Value kernel = convOp.getOperand(1);
    auto kernelType = kernel.getType().dyn_cast<RankedTensorType>();
    if (!kernelType || kernelType.getRank() < 2)
      return false;

    // Depthwise 조건: kernel_shape[1] == 1
    if (kernelType.getShape()[1] != 1)
      return false;

    Value inputX = convOp.getOperand(0);
    auto inputType = inputX.getType().dyn_cast<RankedTensorType>();
    if (!inputType || inputType.getRank() < 2)
      return false;

    int64_t C_in = inputType.getShape()[1]; // 입력 채널 수 (NCHW 형식)

    // Depthwise 조건: group == C_in
    return group == C_in && C_in > 0;
  }

  bool isIntermediateLargerThanDepthwise(
      ONNXConvOp pwConvOp, ONNXConvOp dwConvOp) {
    if (!pwConvOp || !dwConvOp)
      return false;

    Value intermediateTensor = pwConvOp.getResult();
    Value dwResult = dwConvOp.getResult();

    if (!intermediateTensor || intermediateTensor.getType().isa<NoneType>() ||
        !dwResult || dwResult.getType().isa<NoneType>()) {
      return false;
    }

    auto intermediateType = intermediateTensor.getType().dyn_cast<TensorType>();
    auto dwResultType = dwResult.getType().dyn_cast<TensorType>();

    if (!intermediateType || !dwResultType)
      return false;

    return getTensorSize(intermediateType) > getTensorSize(dwResultType);
  }

  // ONNXSplitOp를 생성하고 두 개의 출력 값을 반환하는 함수
  std::pair<Value, Value> createSplitOp(
      OpBuilder &builder, Location loc, Value input, int64_t axis = 0) {
    if (!input || input.getType().isa<NoneType>()) {
      // NoneType 입력이면 None 반환
      Value none = builder.create<ONNXNoneOp>(loc).getResult();
      return {none, none};
    }

    auto inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType || inputType.getRank() <= axis)
      return {nullptr, nullptr}; // invalid shape

    auto shape = inputType.getShape();
    auto elementTy = inputType.getElementType();

    int64_t num_outputs = 2;

    // 총 길이
    int64_t totalSize = shape[axis];
    int64_t firstSize = totalSize / num_outputs;
    int64_t secondSize = totalSize - firstSize;

    // 출력 타입 추정
    SmallVector<int64_t, 4> shapeA(shape.begin(), shape.end());
    shapeA[axis] = firstSize;
    SmallVector<int64_t, 4> shapeB(shape.begin(), shape.end());
    shapeB[axis] = secondSize;

    Type typeA = RankedTensorType::get(shapeA, elementTy);
    Type typeB = RankedTensorType::get(shapeB, elementTy);

    // ONNXSplitOp 생성
    // none type value
    auto noneVal = builder.create<ONNXNoneOp>(loc).getResult();

    auto splitOp =
        builder.create<ONNXSplitOp>(loc, TypeRange{typeA, typeB}, input,
            noneVal, // split 텐서 사용 안 함
            builder.getIntegerAttr(builder.getIntegerType(64, true), axis),
            builder.getIntegerAttr(builder.getIntegerType(64, true),
                num_outputs) // num_outputs 방식 사용
        );

    return {splitOp.getResult(0), splitOp.getResult(1)};
  }

  bool performTensorSplitting(Operation *prePeakOp, OpBuilder &builder) {
    ONNXConvOp pwConv = llvm::dyn_cast<ONNXConvOp>(prePeakOp);
    // isOptimizable 확인으로 인해 pwConv는 유효하다고 가정

    Location loc = pwConv.getLoc();

    // 1. dwConv와 clipOp(optional)를 찾고, 패턴이 적합한지 (단일 사용 경로)
    // 확인
    Value intermediateTensor = pwConv.getResult();
    ONNXClipOp clipOpNode = nullptr;
    ONNXConvOp dwConvNode = nullptr;
    Operation *dwConvOperationForInsertion = nullptr;

    bool patternFound = false;
    if (intermediateTensor.hasOneUse()) {
      Operation *userOfPwConv = *intermediateTensor.getUsers().begin();
      if (auto clipOp = llvm::dyn_cast<ONNXClipOp>(userOfPwConv)) {
        if (clipOp.getResult().hasOneUse()) {
          Operation *userOfClip = *clipOp.getResult().getUsers().begin();
          if (auto dwConv = llvm::dyn_cast<ONNXConvOp>(userOfClip)) {
            if (isDepthwiseConv(dwConv)) {
              dwConvNode = dwConv;
              clipOpNode = clipOp;
              dwConvOperationForInsertion = dwConv;
              patternFound = true;
            }
          }
        }
      } else if (auto dwConv = llvm::dyn_cast<ONNXConvOp>(userOfPwConv)) {
        if (isDepthwiseConv(dwConv)) {
          dwConvNode = dwConv;
          dwConvOperationForInsertion = dwConv;
          patternFound = true;
        }
      }
    }

    if (!patternFound) {
      return false;
    }

    // 2. 피연산자 및 속성 가져오기
    Value X_pw = pwConv.getOperand(0);
    Value W_pw_orig = pwConv.getOperand(1);
    Value B_pw_orig = pwConv.getOperand(2);

    Value W_dw_orig = dwConvNode.getOperand(1);
    Value B_dw_orig = dwConvNode.getOperand(2);

    auto wPwType = W_pw_orig.getType().cast<RankedTensorType>();
    // pwConv 가중치 형태: [M, C_in_pw, 1, 1]
    // M은 pwConv의 출력 채널 수이자 dwConv의 입력 채널 수
    int64_t M = wPwType.getShape()[0];

    // 3. 분할된 가중치와 바이어스 생성
    builder.setInsertionPoint(pwConv);

    auto [W_pw_A, W_pw_B] = createSplitOp(builder, loc, W_pw_orig, 0);
    auto [B_pw_A, B_pw_B] = createSplitOp(builder, loc, B_pw_orig, 0);

    // dwConv 가중치 [M, 1, kH, kW]. 축 0(출력 채널)을 따라 분할
    // dwConv 바이어스 [M]. 축 0을 따라 분할
    auto [W_dw_A, W_dw_B] = createSplitOp(builder, loc, W_dw_orig, 0);
    auto [B_dw_A, B_dw_B] = createSplitOp(builder, loc, B_dw_orig, 0);

    if (!W_pw_A || !W_pw_B || !W_dw_A || !W_dw_B) {
      return false; // 분할 실패
    }

    // 4. 새로운 Conv들의 출력 타입 정의

    // M을 절반으로 분할 (홀수인 경우 두 번째가 더 작음)
    int64_t M_B = M / 2;
    int64_t M_A = M - M_B; // 홀수인 경우 B가 더 작음

    // pwConv 출력 타입 계산
    auto pwOrigOutType = intermediateTensor.getType().cast<RankedTensorType>();
    auto pwShape = pwOrigOutType.getShape();
    auto pwElementType = pwOrigOutType.getElementType();

    SmallVector<int64_t, 4> splitPwShape(pwShape.begin(), pwShape.end());
    splitPwShape[1] = M_A; // 채널 차원 (NCHW에서 C)
    Type splitPwConvOutTypeA =
        RankedTensorType::get(splitPwShape, pwElementType);

    splitPwShape[1] = M_B;
    Type splitPwConvOutTypeB =
        RankedTensorType::get(splitPwShape, pwElementType);

    // dwConv 출력 타입 계산
    auto dwOrigOutType =
        dwConvNode.getResult().getType().cast<RankedTensorType>();
    auto dwShape = dwOrigOutType.getShape();
    auto dwElementType = dwOrigOutType.getElementType();

    SmallVector<int64_t, 4> splitDwShape(dwShape.begin(), dwShape.end());
    splitDwShape[1] = M_A;
    Type splitDwConvOutTypeA =
        RankedTensorType::get(splitDwShape, dwElementType);

    splitDwShape[1] = M_B;
    Type splitDwConvOutTypeB =
        RankedTensorType::get(splitDwShape, dwElementType);

    // 전체 dwConv 출력 타입 (concat 결과)
    Type dwConvOutTypeFull = dwOrigOutType;

    // 속성 가져오기, dwConv의 group 수정 (A, B 각각 다른 그룹 수)
    NamedAttrList pwConvAttrs(pwConv->getAttrs());
    // group attr 타입 맞추기기
    SmallVector<NamedAttribute, 8> newPwConvAttrs;
    for (const NamedAttribute &attr : pwConvAttrs) {
      if (attr.getName().getValue() == "group") {
        auto intAttr = attr.getValue().dyn_cast<IntegerAttr>();
        if (intAttr) {
          // 'group' 속성을 si64 타입으로 명시적으로 생성
          newPwConvAttrs.push_back(builder.getNamedAttr(
              "group", builder.getIntegerAttr(builder.getIntegerType(64, true),
                           intAttr.getSInt())));
        } else {
          newPwConvAttrs.push_back(attr); // 예외 처리
        }
      } else {
        newPwConvAttrs.push_back(attr);
      }
    }
    NamedAttrList dwConvAttrsSplitA(dwConvNode->getAttrs());
    dwConvAttrsSplitA.set(builder.getStringAttr("group"),
        builder.getIntegerAttr(builder.getIntegerType(64, true), M_A));
    NamedAttrList dwConvAttrsSplitB(dwConvNode->getAttrs());
    dwConvAttrsSplitB.set(builder.getStringAttr("group"),
        builder.getIntegerAttr(builder.getIntegerType(64, true), M_B));

    // --- 경로 A 생성 ---
    builder.setInsertionPoint(pwConv);
    ONNXConvOp newPwConvA = builder.create<ONNXConvOp>(loc,
        TypeRange{splitPwConvOutTypeA}, ValueRange{X_pw, W_pw_A, B_pw_A},
        ArrayRef<NamedAttribute>{newPwConvAttrs.begin(), newPwConvAttrs.end()});
    Value interAResult = newPwConvA.getResult();

    if (clipOpNode) {
      ONNXClipOp newClipA = builder.create<ONNXClipOp>(loc, splitPwConvOutTypeA,
          interAResult, clipOpNode.getMin(), clipOpNode.getMax());
      interAResult = newClipA.getResult();
    }

    ONNXConvOp newDwConvA =
        builder.create<ONNXConvOp>(loc, TypeRange{splitDwConvOutTypeA},
            ValueRange{interAResult, W_dw_A, B_dw_A},
            ArrayRef<NamedAttribute>{
                dwConvAttrsSplitA.begin(), dwConvAttrsSplitA.end()});

    // --- 경로 B 생성 ---
    ONNXConvOp newPwConvB = builder.create<ONNXConvOp>(loc,
        TypeRange{splitPwConvOutTypeB}, ValueRange{X_pw, W_pw_B, B_pw_B},
        ArrayRef<NamedAttribute>{pwConvAttrs.begin(), pwConvAttrs.end()});
    Value interBResult = newPwConvB.getResult();

    if (clipOpNode) {
      ONNXClipOp newClipB = builder.create<ONNXClipOp>(loc, splitPwConvOutTypeB,
          interBResult, clipOpNode.getMin(), clipOpNode.getMax());
      interBResult = newClipB.getResult();
    }

    ONNXConvOp newDwConvB =
        builder.create<ONNXConvOp>(loc, TypeRange{splitDwConvOutTypeB},
            ValueRange{interBResult, W_dw_B, B_dw_B},
            ArrayRef<NamedAttribute>{
                dwConvAttrsSplitB.begin(), dwConvAttrsSplitB.end()});

    // 5. 결과 연결
    IntegerAttr concatAxisAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, true), 1); // NCHW 채널 축
    ONNXConcatOp concatOp = builder.create<ONNXConcatOp>(loc, dwConvOutTypeFull,
        ValueRange{newDwConvA.getResult(), newDwConvB.getResult()},
        concatAxisAttr);

    // 6. 기존 연산 제거
    dwConvNode.getResult().replaceAllUsesWith(concatOp.getResult());

    // 종속성을 고려하여 연산 제거
    assert(dwConvNode.getOperation()->use_empty() &&
           "dwConv should have no users");
    dwConvNode.getOperation()->erase();

    if (clipOpNode) {
      assert(clipOpNode.getOperation()->use_empty() &&
             "clipOp should have no users");
      clipOpNode.getOperation()->erase();
    }

    assert(pwConv.getOperation()->use_empty() && "pwConv should have no users");
    pwConv.getOperation()->erase();

    return true;
  }
}; // struct TensorSplittingPass

} // anonymous namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createTensorSplittingPass() {
  return std::make_unique<TensorSplittingPass>();
}
} // namespace onnx_mlir