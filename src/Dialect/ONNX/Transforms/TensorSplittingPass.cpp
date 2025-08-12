#include <fstream>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

// ONNX ops
#include "src/Dialect/ONNX/ONNXOps.hpp"

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

    static bool hasLoggedOnce = false;

    // --- 로깅 로직 ---
    // 이 Pass가 처음 실행될 때 단 한 번만 실행됨.
    if (!hasLoggedOnce) {
      std::ofstream logFile(
          "memory_usage.txt", std::ios::out | std::ios::trunc);
      if (!logFile.is_open()) {
        llvm::errs() << "Error: Could not open memory_usage.txt for writing.\n";
        return;
      }

      llvm::outs()
          << "[TensorSplittingPass] Performing initial memory scan...\n";
      Liveness &liveness = getAnalysis<Liveness>();
      funcOp.walk([&](Operation *op) {
        int64_t currentOpMemoryUsage = calculateMemoryUsageAtOp(op, liveness);
        logFile << currentOpMemoryUsage << "\n";
      });

      logFile.close();
      llvm::outs() << "[TensorSplittingPass] Initial memory usage has been "
                      "written to memory_usage.txt\n";

      // 플래그를 true로 설정하여 다시는 이 블록이 실행되지 않도록 함.
      hasLoggedOnce = true;
    }

    // peak op 분석, 최적화를 반복함. 최적화 불가능하면 루프 종료
    while (1) {
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

      if (!peakOp) {
        llvm::outs()
            << "[TensorSplittingPass] No peak op found or function is empty.\n";
        break;
      }

      Value peakOpOperand = peakOp->getOperand(0);
      prePeakOp = peakOpOperand.getDefiningOp();
      if (!peakOp->getResults().empty() &&
          !peakOp->getResult(0).getUsers().empty()) {
        postPeakOp = *peakOp->getResult(0).getUsers().begin();
      }

      if (!prePeakOp || !postPeakOp) {
        llvm::outs() << "[TensorSplittingPass] Could not determine pre or post "
                        "peak Op. Stopping.\n";
        break;
      }

      llvm::outs() << "[TensorSplittingPass] Peak memory Usage (byte): "
                   << maxMemoryUsage << "\n";
      llvm::outs() << "[TensorSplittingPass] Pre peak op: " << *prePeakOp
                   << "\n";

      if (!isOptimizable(prePeakOp)) {
        llvm::outs() << "[TensorSplittingPass] Peak op is not optimizable.\n";
        break;
      }

      OpBuilder builder(prePeakOp->getContext());

      if (performTensorSplitting(prePeakOp, builder)) {
        static int64_t optCount = 1;
        llvm::outs() << "[TensorSplittingPass] Optimization applied "
                     << optCount << " time" << (optCount == 1 ? "" : "s")
                     << ".\n\n";
        optCount++;
      } else {
        break;
      }
    } // End while
  }

private:
  int64_t getTensorSize(TensorType tensorType) {
    if (!tensorType || !tensorType.hasRank()) {
      return 0;
    }
    Type elementType = tensorType.getElementType();
    unsigned elementBitWidth;
    if (elementType.isIntOrIndex()) {
      elementBitWidth = elementType.getIntOrFloatBitWidth();
    } else if (isa<FloatType>(elementType)) {
      elementBitWidth = cast<FloatType>(elementType).getWidth();
    } else {
      // llvm::errs() << "Unsupported element type: " << elementType << "\n";
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
        return 0;
      totalSize *= dim;
    }
    if (hasDynamicDim)
      return 1;
    return totalSize * (elementBitWidth / 8);
  }

  int64_t calculateMemoryUsageAtOp(Operation *op, Liveness &liveness) {
    const LivenessBlockInfo *blockLiveness =
        liveness.getLiveness(op->getBlock());
    if (!blockLiveness)
      return 0;
    int64_t memoryUsage = 0;
    SmallPtrSet<Value, 16> liveValuesSet;
    for (Value val : blockLiveness->currentlyLiveValues(op)) {
      liveValuesSet.insert(val);
    }
    for (Value result : op->getResults()) {
      liveValuesSet.insert(result);
    }
    for (Value val : liveValuesSet) {
      if (auto tensorType = dyn_cast<TensorType>(val.getType())) {
        memoryUsage += getTensorSize(tensorType);
      }
    }
    return memoryUsage;
  }

  bool isOptimizable(Operation *prePeakOp) {
    if (!isPointwiseConv(prePeakOp)) {
      // llvm::outs() << "[TensorSplittingPass] Memory peak is not at the "
      //                 "pointwise Conv.\n";
      return false;
    }
    auto pwConv = llvm::dyn_cast<ONNXConvOp>(prePeakOp);
    if (!pwConv)
      return false;
    Value intermediateTensor = pwConv.getResult();
    Operation *nextOpAfterPw = nullptr;
    if (!intermediateTensor.hasOneUse()) {
      // llvm::outs()
      //     << "[TensorSplittingPass] Intermediate tensor has multiple
      //     uses.\n";
      return false;
    }
    Operation *user = *intermediateTensor.getUsers().begin();
    if (auto clipOp = llvm::dyn_cast<ONNXClipOp>(user)) {
      if (clipOp.getResult().hasOneUse()) {
        Operation *clipUser = *clipOp.getResult().getUsers().begin();
        if (isDepthwiseConv(clipUser)) {
          nextOpAfterPw = clipUser;
        }
      } else {
        // llvm::outs()
        //     << "[TensorSplittingPass] Intermediate tensor has multiple
        //     uses.\n";
        return false;
      }
    } else if (isDepthwiseConv(user)) {
      nextOpAfterPw = user;
    }
    if (!nextOpAfterPw) {
      // llvm::outs() << "[TensorSplittingPass] No depthwise convolution after "
      //                 "pointwise convolution.\n";
      return false;
    }
    bool larger = isIntermediateLargerThanDepthwise(
        pwConv, llvm::dyn_cast<ONNXConvOp>(nextOpAfterPw));
    if (larger)
      return true;
    else {
      // llvm::outs() << "[TensorSplittingPass] Intermediate tensor is not
      // larger "
      //                 "than depthwise "
      //                 "convolution's result tensor.\n";
      return false;
    }
  }

  bool isPointwiseConv(Operation *op) {
    auto convOp = llvm::dyn_cast_or_null<ONNXConvOp>(op);
    if (!convOp)
      return false;
    Value kernel = convOp.getOperand(1);
    auto kernelType = dyn_cast<RankedTensorType>(kernel.getType());
    if (!kernelType)
      return false;
    ArrayRef<int64_t> shape = kernelType.getShape();
    unsigned rank = shape.size();
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
    auto kernelType = dyn_cast<RankedTensorType>(kernel.getType());
    if (!kernelType || kernelType.getRank() < 2)
      return false;
    if (kernelType.getShape()[1] != 1)
      return false;
    Value inputX = convOp.getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(inputX.getType());
    if (!inputType || inputType.getRank() < 2)
      return false;
    int64_t C_in = inputType.getShape()[1];
    return group == C_in && C_in > 0;
  }

  bool isIntermediateLargerThanDepthwise(
      ONNXConvOp pwConvOp, ONNXConvOp dwConvOp) {
    if (!pwConvOp || !dwConvOp)
      return false;
    Value intermediateTensor = pwConvOp.getResult();
    Value dwResult = dwConvOp.getResult();
    if (!intermediateTensor || isa<NoneType>(intermediateTensor.getType()) ||
        !dwResult || isa<NoneType>(dwResult.getType())) {
      return false;
    }
    auto intermediateType = dyn_cast<TensorType>(intermediateTensor.getType());
    auto dwResultType = dyn_cast<TensorType>(dwResult.getType());
    if (!intermediateType || !dwResultType)
      return false;
    return getTensorSize(intermediateType) > getTensorSize(dwResultType);
  }

  std::pair<Value, Value> createSplitOp(
      OpBuilder &builder, Location loc, Value input, int64_t axis = 0) {
    if (!input || isa<NoneType>(input.getType())) {
      Value none = builder.create<ONNXNoneOp>(loc).getResult();
      return {none, none};
    }
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() <= axis)
      return {nullptr, nullptr};
    auto shape = inputType.getShape();
    auto elementTy = inputType.getElementType();

    int64_t num_outputs = 2;
    int64_t totalSize = shape[axis];
    int64_t firstSize = totalSize / num_outputs;
    int64_t secondSize = totalSize - firstSize;

    SmallVector<int64_t, 4> shapeA(shape.begin(), shape.end());
    shapeA[axis] = firstSize;
    SmallVector<int64_t, 4> shapeB(shape.begin(), shape.end());
    shapeB[axis] = secondSize;

    Type typeA = RankedTensorType::get(shapeA, elementTy);
    Type typeB = RankedTensorType::get(shapeB, elementTy);

    auto noneVal = builder.create<ONNXNoneOp>(loc).getResult();
    auto splitOp = builder.create<ONNXSplitOp>(loc, TypeRange{typeA, typeB},
        input, noneVal,
        builder.getIntegerAttr(builder.getIntegerType(64, true), axis),
        builder.getIntegerAttr(builder.getIntegerType(64, true), num_outputs));
    return {splitOp.getResult(0), splitOp.getResult(1)};
  }

  bool performTensorSplitting(Operation *prePeakOp, OpBuilder &builder) {
    ONNXConvOp pwConv = llvm::dyn_cast<ONNXConvOp>(prePeakOp);
    Location loc = pwConv.getLoc();

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

    Value X_pw = pwConv.getOperand(0);
    Value W_pw_orig = pwConv.getOperand(1);
    Value B_pw_orig = pwConv.getOperand(2);
    Value W_dw_orig = dwConvNode.getOperand(1);
    Value B_dw_orig = dwConvNode.getOperand(2);

    auto wPwType = cast<RankedTensorType>(W_pw_orig.getType());
    int64_t M = wPwType.getShape()[0];

    builder.setInsertionPoint(pwConv);

    auto [W_pw_A, W_pw_B] = createSplitOp(builder, loc, W_pw_orig, 0);
    auto [B_pw_A, B_pw_B] = createSplitOp(builder, loc, B_pw_orig, 0);
    auto [W_dw_A, W_dw_B] = createSplitOp(builder, loc, W_dw_orig, 0);
    auto [B_dw_A, B_dw_B] = createSplitOp(builder, loc, B_dw_orig, 0);
    if (!W_pw_A || !W_pw_B || !W_dw_A || !W_dw_B) {
      return false;
    }

    int64_t M_B = M / 2;
    int64_t M_A = M - M_B;

    auto pwOrigOutType = cast<RankedTensorType>(intermediateTensor.getType());
    auto pwShape = pwOrigOutType.getShape();
    auto pwElementType = pwOrigOutType.getElementType();

    SmallVector<int64_t, 4> splitPwShape(pwShape.begin(), pwShape.end());
    splitPwShape[1] = M_A;
    Type splitPwConvOutTypeA =
        RankedTensorType::get(splitPwShape, pwElementType);
    splitPwShape[1] = M_B;
    Type splitPwConvOutTypeB =
        RankedTensorType::get(splitPwShape, pwElementType);

    auto dwOrigOutType =
        cast<RankedTensorType>(dwConvNode.getResult().getType());
    auto dwShape = dwOrigOutType.getShape();
    auto dwElementType = dwOrigOutType.getElementType();

    SmallVector<int64_t, 4> splitDwShape(dwShape.begin(), dwShape.end());
    splitDwShape[1] = M_A;
    Type splitDwConvOutTypeA =
        RankedTensorType::get(splitDwShape, dwElementType);
    splitDwShape[1] = M_B;
    Type splitDwConvOutTypeB =
        RankedTensorType::get(splitDwShape, dwElementType);

    Type dwConvOutTypeFull = dwOrigOutType;

    NamedAttrList pwConvAttrs(pwConv->getAttrs());
    SmallVector<NamedAttribute, 8> newPwConvAttrs;
    for (const NamedAttribute &attr : pwConvAttrs) {
      if (attr.getName().getValue() == "group") {
        auto intAttr = dyn_cast<IntegerAttr>(attr.getValue());
        if (intAttr) {
          newPwConvAttrs.push_back(builder.getNamedAttr(
              "group", builder.getIntegerAttr(builder.getIntegerType(64, true),
                           intAttr.getSInt())));
        } else {
          newPwConvAttrs.push_back(attr);
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

    IntegerAttr concatAxisAttr =
        builder.getIntegerAttr(builder.getIntegerType(64, true), 1);

    ONNXConcatOp concatOp = builder.create<ONNXConcatOp>(loc, dwConvOutTypeFull,
        ValueRange{newDwConvA.getResult(), newDwConvB.getResult()},
        concatAxisAttr);

    dwConvNode.getResult().replaceAllUsesWith(concatOp.getResult());
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
