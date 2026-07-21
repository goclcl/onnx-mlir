//===------ PeakMemOptRules.cpp - op별 분할 규칙 구현 ------===//
//
// PeakMemOptRules.hpp의 구현. 각 규칙은 "분할된 값이 이 op를 통과하면 무슨
// 일이 일어나는가"를 순수 함수로 서술할 뿐, IR은 만들지 않는다(patchClone의
// attribute/보조 상수 보정만 예외).
//
//===----------------------------------------------------------------===//

#include "src/Dialect/ONNX/Transforms/PeakMemOptRules.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/PeakMemOptUtils.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace peakmem {

namespace {

//===--------------------- 공용 헬퍼 ---------------------===//

// "상수로 취급 가능한 값"인가. 엔진이 앞선 재작성에서 만든 상수의 Split
// 조각도 오프라인으로 접히므로 상수와 동일하게 본다 — 엔진 산출 conv 등을
// 다시 weight-split할 수 있게 하는 핵심 조건.
bool isConstantValue(Value v) { return isFoldedOffline(v); }

bool isNoneValue(Value v) { return isa<NoneType>(v.getType()); }

// ranked shape를 얻는다. unranked/스칼라 미지원 경로는 호출부에서 실패 처리.
std::optional<ArrayRef<int64_t>> getShapeOf(Value v) {
  auto rtt = dyn_cast<RankedTensorType>(v.getType());
  if (!rtt || !rtt.hasStaticShape())
    return std::nullopt;
  return rtt.getShape();
}

//===--------------------- Conv ---------------------===//
// 다루는 상황:
//  - planWeightSplit: group==1 conv의 out-channel 분할 (W ax0, B ax0).
//  - propagate:
//    * 입력이 Partitioned(1)(channel):
//        depthwise(group==C_in)  → 채널 파티션 통과 + W/B ax0 분할,
//                                  patchClone에서 group 재계산.
//        group==1               → 입력채널이 수축되므로 result는 Reduced.
//                                  W ax1 분할, bias는 ApplyOnce.
//    * spatial(2,3)은 halo(경계 겹침) 처리가 없어 불가.

struct ConvRule final : public SplitRule {
  FailureOr<OpSplitStep> propagate(Operation *op, ArrayRef<SplitState> operands,
      int nBranches) const override {
    auto conv = dyn_cast<ONNXConvOp>(op);
    if (!conv || operands.size() < 2)
      return failure();
    // weight/bias가 분할 대상 상태로 들어오는 경우는 다루지 않는다.
    for (unsigned i = 1; i < operands.size(); ++i)
      if (!operands[i].isUntouched())
        return failure();

    const SplitState &x = operands[0];
    if (x.isReduced())
      return failure();
    if (x.isUntouched()) {
      OpSplitStep step;
      step.results.push_back(SplitState::untouched());
      return step;
    }

    auto xShape = getShapeOf(conv.getX());
    if (!xShape)
      return failure();

    // Conv 입력은 NCHW 고정이므로 축 0은 항상 batch(분할 금지 정책),
    // 축 2/3(spatial)은 halo(경계 겹침) 처리가 없어 미지원.
    if (x.axis != 1)
      return failure();

    int64_t group = conv.getGroup();
    int64_t inChannels = (*xShape)[1];

    OpSplitStep step;
    if (group == inChannels && inChannels > 1) {
      // depthwise: 채널별 독립 연산이라 파티션이 그대로 통과한다.
      step.results.push_back(SplitState::partitioned(1));
      step.slices.push_back({1, 0, SliceReq::Kind::Partition});
      if (!isNoneValue(conv.getB()))
        step.slices.push_back({2, 0, SliceReq::Kind::Partition});
      return step;
    }
    if (group == 1) {
      // 입력채널이 수축(reduction) 축: 브랜치 결과는 부분합이 된다.
      step.results.push_back(SplitState::reduced());
      step.slices.push_back({1, 1, SliceReq::Kind::Partition});
      if (!isNoneValue(conv.getB()))
        step.slices.push_back({2, 0, SliceReq::Kind::ApplyOnce});
      return step;
    }
    return failure(); // grouped conv(1<group<C)는 미지원
  }

  FailureOr<OpSplitStep> planWeightSplit(
      Operation *op, int nBranches) const override {
    auto conv = dyn_cast<ONNXConvOp>(op);
    if (!conv)
      return failure();
    if (conv.getGroup() != 1)
      return failure(); // depthwise/grouped의 out-channel 분할은 미지원
    if (!isConstantValue(conv.getW()))
      return failure();
    auto wShape = getShapeOf(conv.getW());
    if (!wShape || (*wShape)[0] < nBranches)
      return failure();

    OpSplitStep step;
    step.results.push_back(SplitState::partitioned(1)); // 출력 채널 파티션
    step.slices.push_back({1, 0, SliceReq::Kind::Partition});
    if (!isNoneValue(conv.getB()))
      step.slices.push_back({2, 0, SliceReq::Kind::Partition});
    return step;
  }

  void patchClone(OpBuilder &builder, Operation *cloned, int branchIdx,
      int nBranches, const OpSplitStep &step) const override {
    // depthwise 통과(채널 파티션)의 경우 group을 조각의 채널 수로 재계산.
    auto conv = cast<ONNXConvOp>(cloned);
    if (step.results.empty() || !step.results[0].isPartitioned() ||
        step.results[0].axis != 1)
      return;
    auto wShape = getShapeOf(conv.getW());
    if (!wShape)
      return;
    int64_t origGroup = conv.getGroup();
    if (origGroup <= 1)
      return; // group==1은 보정 불필요
    auto si64Ty =
        IntegerType::get(builder.getContext(), 64, IntegerType::Signed);
    conv.setGroupAttr(IntegerAttr::get(si64Ty, (*wShape)[0]));
  }
};

//===--------------------- MatMul ---------------------===//
// numpy 스타일 broadcast matmul: out[..., M, N] = A[..., M, K] @ B[..., K, N].
// A의 마지막 축 = K, B의 뒤에서 두 번째 축 = K.
//  - planWeightSplit: 상수 operand의 M(또는 N) 축을 자르면 출력의 해당 축이
//    파티션된다. K 축을 자르는 계획은 내지 않는다(출력이 시작부터 부분합).
//  - propagate: activation의 파티션 축이 K로 들어오면 result는 Reduced가 되고
//    반대편 operand도 같은 K 조각을 갖도록 요구한다.

struct MatMulRule final : public SplitRule {
  FailureOr<OpSplitStep> propagate(Operation *op, ArrayRef<SplitState> operands,
      int nBranches) const override {
    auto mm = dyn_cast<ONNXMatMulOp>(op);
    if (!mm || operands.size() != 2)
      return failure();
    auto aShape = getShapeOf(mm.getA());
    auto bShape = getShapeOf(mm.getB());
    auto outShape = getShapeOf(mm.getY());
    if (!aShape || !bShape || !outShape)
      return failure();
    int64_t rA = aShape->size(), rB = bShape->size(), rO = outShape->size();
    if (rA < 2 || rB < 2)
      return failure();

    const SplitState &sA = operands[0], &sB = operands[1];
    if (sA.isReduced() || sB.isReduced())
      return failure();

    OpSplitStep step;
    // --- A가 파티션된 경우 ---
    if (sA.isPartitioned() && sB.isUntouched()) {
      if (sA.axis == rA - 1) {
        // K 분할: 결과는 부분합. B도 K(뒤에서 두 번째)로 잘라야 한다.
        step.results.push_back(SplitState::reduced());
        if (isConstantValue(mm.getB()))
          step.slices.push_back(
              {1, static_cast<int64_t>(rB - 2), SliceReq::Kind::Partition});
        else
          return failure(); // 양쪽 activation의 K 동시분할은 seed로만 가능
        return step;
      }
      // M 축(rA-2)은 B와 무관하므로 그대로 통과. stack 축(d < rA-2)은 반대편
      // B가 같은 stack 축을 실제로 가지면(크기>1) B도 함께 잘라야 하는
      // 경우인데, 검증된 적 없는 경로이므로 안전하게 기각한다.
      if (sA.axis < rA - 2) {
        int64_t bAxis = sA.axis - (rA - rB);
        if (bAxis >= 0 && (*bShape)[bAxis] > 1)
          return failure();
      }
      step.results.push_back(SplitState::partitioned(sA.axis + (rO - rA)));
      return step;
    }
    // --- B가 파티션된 경우 ---
    if (sB.isPartitioned() && sA.isUntouched()) {
      if (sB.axis == rB - 2) {
        // K 분할: A도 K(마지막 축)로 잘라야 한다.
        step.results.push_back(SplitState::reduced());
        if (isConstantValue(mm.getA()))
          step.slices.push_back(
              {0, static_cast<int64_t>(rA - 1), SliceReq::Kind::Partition});
        else
          return failure();
        return step;
      }
      if (sB.axis == rB - 1) { // N 축
        step.results.push_back(SplitState::partitioned(rO - 1));
        return step;
      }
      // B의 stack 축: 반대편 A가 같은 stack 축을 실제로 가지면(크기>1)
      // A도 함께 잘라야 하는 경우인데, 검증된 적 없는 경로이므로 안전하게
      // 기각한다.
      {
        int64_t aAxis = sB.axis - (rB - rA);
        if (aAxis >= 0 && (*aShape)[aAxis] > 1)
          return failure();
      }
      step.results.push_back(SplitState::partitioned(sB.axis + (rO - rB)));
      return step;
    }
    // --- 양쪽 다 파티션: 같은 출력 축으로 정렬될 때만 허용 ---
    if (sA.isPartitioned() && sB.isPartitioned()) {
      if (sA.axis == rA - 1 && sB.axis == rB - 2) { // 둘 다 K
        step.results.push_back(SplitState::reduced());
        return step;
      }
      int64_t oA = sA.axis + (rO - rA), oB = sB.axis + (rO - rB);
      if (sA.axis < rA - 2 && sB.axis < rB - 2 && oA == oB) { // 같은 batch 축
        step.results.push_back(SplitState::partitioned(oA));
        return step;
      }
      return failure();
    }
    // 둘 다 Untouched
    step.results.push_back(SplitState::untouched());
    return step;
  }

  FailureOr<OpSplitStep> planWeightSplit(
      Operation *op, int nBranches) const override {
    auto mm = dyn_cast<ONNXMatMulOp>(op);
    if (!mm)
      return failure();
    auto aShape = getShapeOf(mm.getA());
    auto bShape = getShapeOf(mm.getB());
    auto outShape = getShapeOf(mm.getY());
    if (!aShape || !bShape || !outShape)
      return failure();
    int64_t rA = aShape->size(), rB = bShape->size(), rO = outShape->size();
    if (rA < 2 || rB < 2)
      return failure();

    bool aConst = isConstantValue(mm.getA());
    bool bConst = isConstantValue(mm.getB());
    OpSplitStep step;
    if (bConst && !aConst && (*bShape)[rB - 1] >= nBranches) {
      // act @ W: W의 N 축을 자르면 출력 마지막 축이 파티션.
      step.results.push_back(SplitState::partitioned(rO - 1));
      step.slices.push_back(
          {1, static_cast<int64_t>(rB - 1), SliceReq::Kind::Partition});
      return step;
    }
    if (aConst && !bConst && (*aShape)[rA - 2] >= nBranches) {
      // W @ act: W의 M 축을 자르면 출력 뒤에서 두 번째 축이 파티션.
      step.results.push_back(SplitState::partitioned(rO - 2));
      step.slices.push_back(
          {0, static_cast<int64_t>(rA - 2), SliceReq::Kind::Partition});
      return step;
    }
    return failure();
  }
};

//===--------------------- Reshape ---------------------===//
// 파티션된 축이 병합/분해되지 않고 크기 그대로 살아남는 경우에만 통과를
// 허용한다: 입력 축 d 앞쪽 원소 수 곱 == 출력 축 j 앞쪽 원소 수 곱이고
// 두 축의 크기가 같으면 d → j 로 리매핑. shape 상수는 patchClone이 브랜치
// 조각 크기로 다시 쓴다(균등 분할만 지원).

struct ReshapeRule final : public SplitRule {
  FailureOr<OpSplitStep> propagate(Operation *op, ArrayRef<SplitState> operands,
      int nBranches) const override {
    auto reshape = dyn_cast<ONNXReshapeOp>(op);
    if (!reshape || operands.empty())
      return failure();
    const SplitState &data = operands[0];
    if (data.isReduced())
      return failure();
    OpSplitStep step;
    if (data.isUntouched()) {
      step.results.push_back(SplitState::untouched());
      return step;
    }
    // patchClone이 shape 상수의 값을 직접 읽어 재작성하므로, 여기서는
    // 진짜 ONNXConstantOp만 허용한다 (Split 조각 등은 값을 읽을 수 없음).
    if (!reshape.getShape().getDefiningOp<ONNXConstantOp>())
      return failure();

    auto inShape = getShapeOf(reshape.getData());
    auto outShape = getShapeOf(reshape.getReshaped());
    if (!inShape || !outShape)
      return failure();
    int64_t d = data.axis;
    if (d < 0 || d >= (int64_t)inShape->size())
      return failure();
    int64_t targetSize = (*inShape)[d];
    if (targetSize % nBranches != 0)
      return failure(); // shape 상수 재작성이 균등 분할만 지원

    int64_t inProd = 1;
    for (int64_t i = 0; i < d; ++i)
      inProd *= (*inShape)[i];

    int64_t outProd = 1;
    for (int64_t j = 0; j < (int64_t)outShape->size(); ++j) {
      if ((*outShape)[j] == targetSize && outProd == inProd) {
        step.results.push_back(SplitState::partitioned(j));
        return step;
      }
      outProd *= (*outShape)[j];
    }
    return failure(); // 파티션 축이 reshape로 쪼개지거나 합쳐짐
  }

  void patchClone(OpBuilder &builder, Operation *cloned, int branchIdx,
      int nBranches, const OpSplitStep &step) const override {
    auto reshape = cast<ONNXReshapeOp>(cloned);
    if (step.results.empty() || !step.results[0].isPartitioned())
      return;
    int64_t j = step.results[0].axis;
    auto constOp = reshape.getShape().getDefiningOp<ONNXConstantOp>();
    if (!constOp)
      return;
    auto dense = dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
    if (!dense)
      return;
    SmallVector<int64_t, 4> vals(dense.getValues<int64_t>());
    if (j < 0 || j >= (int64_t)vals.size())
      return;
    // ONNX Reshape shape 항목의 특수값은 손대지 않는다:
    //  -1 = 자동 추론 dim — 잘린 데이터로부터 재추론되어 알아서 절반이 된다.
    //   0 = 입력 dim 복사(allowzero=0) — 입력이 잘리므로 복사값도 따라 잘린다.
    if (vals[j] == -1 || vals[j] == 0)
      return;
    if (vals[j] % nBranches != 0)
      return;
    vals[j] /= nBranches;

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(cloned);
    auto ty = RankedTensorType::get(
        {(int64_t)vals.size()}, builder.getI64Type());
    Value newShape =
        builder
            .create<ONNXConstantOp>(cloned->getLoc(), Attribute(),
                DenseElementsAttr::get(ty, ArrayRef<int64_t>(vals)))
            .getResult();
    reshape.setOperand(1, newShape);
  }
};

//===--------------------- Transpose ---------------------===//
// 파티션 축이 perm에 따라 자리만 옮겨 그대로 통과한다. 자를 것도 보정할
// 것도 없다.

struct TransposeRule final : public SplitRule {
  // perm 배열을 얻는다 (attr이 없으면 ONNX 기본값 = 축 순서 뒤집기).
  static SmallVector<int64_t, 4> getPerm(ONNXTransposeOp op, int64_t rank) {
    SmallVector<int64_t, 4> perm;
    if (auto permAttr = op.getPermAttr()) {
      for (Attribute a : permAttr)
        perm.push_back(cast<IntegerAttr>(a).getInt());
    } else {
      for (int64_t i = rank - 1; i >= 0; --i)
        perm.push_back(i);
    }
    return perm;
  }

  FailureOr<OpSplitStep> propagate(Operation *op, ArrayRef<SplitState> operands,
      int nBranches) const override {
    auto transpose = dyn_cast<ONNXTransposeOp>(op);
    if (!transpose || operands.empty())
      return failure();
    const SplitState &data = operands[0];
    if (data.isReduced())
      return failure();
    OpSplitStep step;
    if (data.isUntouched()) {
      step.results.push_back(SplitState::untouched());
      return step;
    }
    auto inShape = getShapeOf(transpose.getData());
    if (!inShape)
      return failure();
    int64_t rank = inShape->size();
    SmallVector<int64_t, 4> perm = getPerm(transpose, rank);
    for (int64_t j = 0; j < rank; ++j) {
      if (perm[j] == data.axis) {
        step.results.push_back(SplitState::partitioned(j));
        return step;
      }
    }
    return failure();
  }
};

//===--------------------- LayerNormalization ---------------------===//
// 정규화는 axis 이후의 축들에서 일어난다. 그보다 앞선 축의 파티션은 정규화
// 그룹들을 통째로 나누는 것이라 그대로 통과하며, scale/bias는 정규화 축
// 위의 값이므로 건드릴 필요 없다. 정규화되는 축의 파티션은 통계가 조각마다
// 달라지므로 불가.

struct LayerNormRule final : public SplitRule {
  FailureOr<OpSplitStep> propagate(Operation *op, ArrayRef<SplitState> operands,
      int nBranches) const override {
    auto ln = dyn_cast<ONNXLayerNormalizationOp>(op);
    if (!ln || operands.empty())
      return failure();
    // scale/bias가 분할 대상 상태로 들어오는 경우는 다루지 않는다.
    for (unsigned i = 1; i < operands.size(); ++i)
      if (!operands[i].isUntouched())
        return failure();
    const SplitState &x = operands[0];
    if (x.isReduced())
      return failure();

    auto xShape = getShapeOf(ln.getX());
    if (!xShape)
      return failure();
    int64_t rank = xShape->size();
    int64_t normAxis = ln.getAxis();
    if (normAxis < 0)
      normAxis += rank;

    OpSplitStep step;
    if (x.isUntouched()) {
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        step.results.push_back(SplitState::untouched());
      return step;
    }
    if (x.axis >= normAxis)
      return failure(); // 정규화되는 축은 나눌 수 없다

    // Y와 (있다면) Mean/InvStdDev 모두 앞쪽 축을 공유하므로 같은 축으로
    // 파티션된다. NoneType 결과는 분할과 무관.
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      if (isa<NoneType>(op->getResult(i).getType()))
        step.results.push_back(SplitState::untouched());
      else
        step.results.push_back(SplitState::partitioned(x.axis));
    }
    return step;
  }
};

//===--------------------- Softmax ---------------------===//
// softmax는 지정된 한 축 안에서만 정규화하므로, 그 축이 아닌 어떤 축의
// 파티션도 값 변화 없이 통과한다 (LayerNorm과 달리 축 뒤쪽도 허용).
// rank가 보존되므로 axis attr(양수/음수 모두)는 보정 없이 유효하다.
// 검증: vit 1x12x197x197(axis=-1)을 head축(1)·query축(2)으로 수동 분할해
// bit-exact 확인 (2026-07-20).

struct SoftmaxRule final : public SplitRule {
  FailureOr<OpSplitStep> propagate(Operation *op, ArrayRef<SplitState> operands,
      int nBranches) const override {
    auto sm = dyn_cast<ONNXSoftmaxOp>(op);
    if (!sm || operands.empty())
      return failure();
    const SplitState &x = operands[0];
    if (x.isReduced())
      return failure();
    OpSplitStep step;
    if (x.isUntouched()) {
      step.results.push_back(SplitState::untouched());
      return step;
    }
    auto xShape = getShapeOf(sm.getInput());
    if (!xShape)
      return failure();
    int64_t rank = xShape->size();
    int64_t smAxis = sm.getAxis();
    if (smAxis < 0)
      smAxis += rank;
    if (x.axis == smAxis)
      return failure(); // 정규화되는 축은 나눌 수 없다
    step.results.push_back(SplitState::partitioned(x.axis));
    return step;
  }
};

//===--------------------- MaxPool ---------------------===//
// 채널(1) 파티션만 통과. spatial은 커널 겹침(halo) 미지원.

struct MaxPoolRule final : public SplitRule {
  FailureOr<OpSplitStep> propagate(Operation *op, ArrayRef<SplitState> operands,
      int nBranches) const override {
    if (!isa<ONNXMaxPoolSingleOutOp>(op) || operands.empty())
      return failure();
    const SplitState &x = operands[0];
    if (x.isReduced())
      return failure();
    OpSplitStep step;
    if (x.isUntouched()) {
      step.results.push_back(SplitState::untouched());
      return step;
    }
    // NCHW 고정: 축 0은 batch(분할 금지 정책), spatial은 커널 겹침 미지원.
    if (x.axis == 1) {
      step.results.push_back(SplitState::partitioned(1));
      return step;
    }
    return failure();
  }
};

//===--------------------- Elementwise (카테고리) ---------------------===//

// 단항: 데이터 operand(0번)의 분할 상태가 그대로 통과. 나머지 operand
// (Clip의 min/max 등)는 분할과 무관해야 한다.
struct EltwiseUnaryRule final : public SplitRule {
  FailureOr<OpSplitStep> propagate(Operation *op, ArrayRef<SplitState> operands,
      int nBranches) const override {
    if (operands.empty() || operands[0].isReduced())
      return failure();
    for (unsigned i = 1; i < operands.size(); ++i)
      if (!operands[i].isUntouched())
        return failure();
    OpSplitStep step;
    step.results.push_back(operands[0]);
    return step;
  }
};

// 이항(broadcast): 두 operand의 분할 상태가 출력의 같은 축으로 정렬되어야
// 한다. 한쪽이 분할과 무관하면 그 쪽의 대응 축을 검사한다:
//   대응 축이 없거나 크기 1  → broadcast, 그대로 사용
//   크기 > 1 인 상수         → 조각으로 잘라 사용 (SliceReq)
//   크기 > 1 인 activation   → 잘라 줄 방법이 없으므로 실패
struct EltwiseBinaryRule final : public SplitRule {
  FailureOr<OpSplitStep> propagate(Operation *op, ArrayRef<SplitState> operands,
      int nBranches) const override {
    if (operands.size() != 2 || op->getNumResults() != 1)
      return failure();
    const SplitState &s0 = operands[0], &s1 = operands[1];
    if (s0.isReduced() || s1.isReduced())
      return failure();

    auto outShape = getShapeOf(op->getResult(0));
    if (!outShape)
      return failure();
    int64_t rO = outShape->size();

    OpSplitStep step;
    if (s0.isUntouched() && s1.isUntouched()) {
      step.results.push_back(SplitState::untouched());
      return step;
    }

    auto outAxisOf = [&](unsigned idx,
                         const SplitState &s) -> std::optional<int64_t> {
      auto shape = getShapeOf(op->getOperand(idx));
      if (!shape)
        return std::nullopt;
      return s.axis + (rO - (int64_t)shape->size());
    };

    if (s0.isPartitioned() && s1.isPartitioned()) {
      auto o0 = outAxisOf(0, s0), o1 = outAxisOf(1, s1);
      if (!o0 || !o1 || *o0 != *o1)
        return failure();
      step.results.push_back(SplitState::partitioned(*o0));
      return step;
    }

    unsigned pIdx = s0.isPartitioned() ? 0 : 1; // 파티션된 쪽
    unsigned uIdx = 1 - pIdx;                   // 무관한 쪽
    auto oAxis = outAxisOf(pIdx, operands[pIdx]);
    if (!oAxis)
      return failure();
    step.results.push_back(SplitState::partitioned(*oAxis));

    auto uShape = getShapeOf(op->getOperand(uIdx));
    if (!uShape)
      return failure();
    int64_t uAxis = *oAxis - (rO - (int64_t)uShape->size());
    if (uAxis < 0 || (*uShape)[uAxis] == 1)
      return step; // broadcast: 건드릴 필요 없음
    if (isConstantValue(op->getOperand(uIdx))) {
      step.slices.push_back({uIdx, uAxis, SliceReq::Kind::Partition});
      return step;
    }
    return failure(); // 분할되지 않은 activation과는 정렬 불가
  }
};

bool isEltwiseUnary(Operation *op) {
  return isa<ONNXReluOp, ONNXSigmoidOp, ONNXGeluOp, ONNXClipOp, ONNXTanhOp>(
      op);
}

bool isEltwiseBinary(Operation *op) {
  return isa<ONNXAddOp, ONNXMulOp, ONNXSubOp, ONNXDivOp>(op);
}

} // namespace

//===--------------------- Registry ---------------------===//

const SplitRuleRegistry &SplitRuleRegistry::instance() {
  static SplitRuleRegistry theRegistry;
  return theRegistry;
}

SplitRuleRegistry::SplitRuleRegistry() {
  static ConvRule convRule;
  static MatMulRule matMulRule;
  static ReshapeRule reshapeRule;
  static MaxPoolRule maxPoolRule;
  static TransposeRule transposeRule;
  static LayerNormRule layerNormRule;
  static SoftmaxRule softmaxRule;
  static EltwiseUnaryRule unaryRule;
  static EltwiseBinaryRule binaryRule;

  exactRules["onnx.Conv"] = &convRule;
  exactRules["onnx.MatMul"] = &matMulRule;
  exactRules["onnx.Reshape"] = &reshapeRule;
  exactRules["onnx.MaxPoolSingleOut"] = &maxPoolRule;
  exactRules["onnx.Transpose"] = &transposeRule;
  exactRules["onnx.LayerNormalization"] = &layerNormRule;
  exactRules["onnx.Softmax"] = &softmaxRule;
  eltwiseUnaryRule = &unaryRule;
  eltwiseBinaryRule = &binaryRule;
}

const SplitRule *SplitRuleRegistry::lookup(Operation *op) const {
  if (const SplitRule *rule =
          exactRules.lookup(op->getName().getStringRef()))
    return rule;
  if (isEltwiseUnary(op))
    return eltwiseUnaryRule;
  if (isEltwiseBinary(op))
    return eltwiseBinaryRule;
  return nullptr;
}

} // namespace peakmem
} // namespace onnx_mlir
