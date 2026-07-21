//===------ TinyMoUtils.cpp - TinyMo 공용 유틸/엔진 구현 ------===//

#include "src/Dialect/ONNX/Transforms/TinyMoUtils.hpp"

#include <algorithm>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace onnx_mlir {
namespace tinymo {

int64_t getTensorSize(TensorType tensorType) {
  if (!tensorType || !tensorType.hasRank())
    return 0;
  Type elementType = tensorType.getElementType();
  unsigned elementBitWidth;
  if (elementType.isIntOrIndex()) {
    elementBitWidth = elementType.getIntOrFloatBitWidth();
  } else if (isa<FloatType>(elementType)) {
    elementBitWidth = cast<FloatType>(elementType).getWidth();
  } else {
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

int64_t memoryUsageAtOp(
    Operation *op, Liveness &liveness, bool includeConstants) {
  const LivenessBlockInfo *blockLiveness = liveness.getLiveness(op->getBlock());
  if (!blockLiveness)
    return 0;
  int64_t memoryUsage = 0;
  SmallPtrSet<Value, 16> liveValuesSet;
  for (Value val : blockLiveness->currentlyLiveValues(op))
    liveValuesSet.insert(val);
  for (Value result : op->getResults())
    liveValuesSet.insert(result);
  for (Value val : liveValuesSet) {
    if (!includeConstants) {
      if (Operation *def = val.getDefiningOp())
        if (isa<ONNXConstantOp>(def))
          continue;
    }
    if (auto tensorType = dyn_cast<TensorType>(val.getType()))
      memoryUsage += getTensorSize(tensorType);
  }
  return memoryUsage;
}

//===--------------------- tensor splitting ---------------------===//

namespace {

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

// input의 정의가 anchor(모든 새 op가 삽입되는 지점) 앞에 오도록 보장한다.
// 재분할 시 이전 반복이 만든 가중치 Split이 anchor보다 아래에 있을 수
// 있는데, 그 체인은 상수/None에서만 유도되므로 통째로 끌어올려도 안전하다.
void ensureDefDominates(Value v, Operation *anchor) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != anchor->getBlock())
    return;
  if (def->isBeforeInBlock(anchor))
    return;
  for (Value opnd : def->getOperands())
    ensureDefDominates(opnd, anchor);
  def->moveBefore(anchor);
}

std::pair<Value, Value> createSplitOp(OpBuilder &builder, Location loc,
    Value input, Operation *anchor, int64_t axis = 0) {
  if (input && anchor)
    ensureDefDominates(input, anchor);
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

} // namespace

std::optional<SplitCandidate> findSplitCandidate(Operation *peakOp) {
  if (!peakOp || peakOp->getNumOperands() == 0)
    return std::nullopt;
  Operation *prePeakOp = peakOp->getOperand(0).getDefiningOp();
  if (!prePeakOp || !isPointwiseConv(prePeakOp))
    return std::nullopt;
  auto pwConv = llvm::dyn_cast<ONNXConvOp>(prePeakOp);
  if (!pwConv)
    return std::nullopt;

  Value intermediateTensor = pwConv.getResult();
  if (!intermediateTensor.hasOneUse())
    return std::nullopt;

  SplitCandidate cand;
  cand.pwConv = pwConv;
  Operation *user = *intermediateTensor.getUsers().begin();
  if (auto clipOp = llvm::dyn_cast<ONNXClipOp>(user)) {
    if (!clipOp.getResult().hasOneUse())
      return std::nullopt;
    Operation *clipUser = *clipOp.getResult().getUsers().begin();
    if (!isDepthwiseConv(clipUser))
      return std::nullopt;
    cand.clip = clipOp;
    cand.dwConv = llvm::cast<ONNXConvOp>(clipUser);
  } else if (isDepthwiseConv(user)) {
    cand.dwConv = llvm::cast<ONNXConvOp>(user);
  } else {
    return std::nullopt;
  }

  auto interType = dyn_cast<TensorType>(intermediateTensor.getType());
  auto dwType = dyn_cast<TensorType>(cand.dwConv.getResult().getType());
  if (!interType || !dwType)
    return std::nullopt;
  cand.interBytes = getTensorSize(interType);
  // 중간 텐서가 dw 출력보다 클 때만 분할이 이득이다.
  if (cand.interBytes <= getTensorSize(dwType))
    return std::nullopt;
  return cand;
}

int64_t splitExpectedReduction(const SplitCandidate &cand) {
  return cand.interBytes / 2;
}

bool applyTensorSplitting(SplitCandidate &cand, OpBuilder &builder) {
  ONNXConvOp pwConv = cand.pwConv;
  ONNXClipOp clipOpNode = cand.clip;
  ONNXConvOp dwConvNode = cand.dwConv;
  Location loc = pwConv.getLoc();
  Value intermediateTensor = pwConv.getResult();

  Value X_pw = pwConv.getOperand(0);
  Value W_pw_orig = pwConv.getOperand(1);
  Value B_pw_orig = pwConv.getOperand(2);
  Value W_dw_orig = dwConvNode.getOperand(1);
  Value B_dw_orig = dwConvNode.getOperand(2);

  auto wPwType = cast<RankedTensorType>(W_pw_orig.getType());
  int64_t M = wPwType.getShape()[0];

  builder.setInsertionPoint(pwConv);

  Operation *anchor = pwConv.getOperation();
  auto [W_pw_A, W_pw_B] = createSplitOp(builder, loc, W_pw_orig, anchor, 0);
  auto [B_pw_A, B_pw_B] = createSplitOp(builder, loc, B_pw_orig, anchor, 0);
  auto [W_dw_A, W_dw_B] = createSplitOp(builder, loc, W_dw_orig, anchor, 0);
  auto [B_dw_A, B_dw_B] = createSplitOp(builder, loc, B_dw_orig, anchor, 0);
  if (!W_pw_A || !W_pw_B || !W_dw_A || !W_dw_B)
    return false;

  int64_t M_B = M / 2;
  int64_t M_A = M - M_B;

  auto pwOrigOutType = cast<RankedTensorType>(intermediateTensor.getType());
  auto pwShape = pwOrigOutType.getShape();
  auto pwElementType = pwOrigOutType.getElementType();

  SmallVector<int64_t, 4> splitPwShape(pwShape.begin(), pwShape.end());
  splitPwShape[1] = M_A;
  Type splitPwConvOutTypeA = RankedTensorType::get(splitPwShape, pwElementType);
  splitPwShape[1] = M_B;
  Type splitPwConvOutTypeB = RankedTensorType::get(splitPwShape, pwElementType);

  auto dwOrigOutType = cast<RankedTensorType>(dwConvNode.getResult().getType());
  auto dwShape = dwOrigOutType.getShape();
  auto dwElementType = dwOrigOutType.getElementType();

  SmallVector<int64_t, 4> splitDwShape(dwShape.begin(), dwShape.end());
  splitDwShape[1] = M_A;
  Type splitDwConvOutTypeA = RankedTensorType::get(splitDwShape, dwElementType);
  splitDwShape[1] = M_B;
  Type splitDwConvOutTypeB = RankedTensorType::get(splitDwShape, dwElementType);

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

//===--------------------- tensor spilling ---------------------===//

std::optional<SpillCandidate> findSpillCandidate(
    func::FuncOp funcOp, Liveness &liveness, Operation *peakOp) {
  Block *block = peakOp->getBlock();
  const LivenessBlockInfo *blockLiveness = liveness.getLiveness(block);
  if (!blockLiveness)
    return std::nullopt;

  // op → 블록 내 위치 인덱스
  llvm::DenseMap<Operation *, int64_t> opIndex;
  int64_t idx = 0;
  for (Operation &op : *block)
    opIndex[&op] = idx++;
  int64_t peakIdx = opIndex.lookup(peakOp);

  SmallPtrSet<Value, 16> liveValues;
  for (Value v : blockLiveness->currentlyLiveValues(peakOp))
    liveValues.insert(v);

  std::optional<SpillCandidate> best;
  for (Value v : liveValues) {
    Operation *def = v.getDefiningOp();
    // 블록 인자(모델 입력)와 상수(파라미터, 플래시 상주)는 대상이 아니다.
    if (!def || isa<ONNXConstantOp>(def) || isa<ONNXNoneOp>(def))
      continue;
    // 우리가 삽입한 spill의 토큰(1바이트)은 cold range가 정의상 가장 길어
    // 다시 후보로 뽑히면 무한 재귀한다 — 자체 산출물은 제외.
    if (auto custom = dyn_cast<ONNXCustomOp>(def))
      if (custom.getFunctionName() == "om_spill")
        continue;
    if (def->getBlock() != block)
      continue;
    auto tensorType = dyn_cast<TensorType>(v.getType());
    if (!tensorType)
      continue;
    int64_t bytes = getTensorSize(tensorType);
    if (bytes <= 0)
      continue;

    // 접근 지점: 정의 + 모든 사용
    SmallVector<int64_t, 8> accesses;
    accesses.push_back(opIndex.lookup(def));
    bool sameBlock = true;
    for (Operation *user : v.getUsers()) {
      if (user->getBlock() != block) {
        sameBlock = false;
        break;
      }
      accesses.push_back(opIndex.lookup(user));
    }
    if (!sameBlock)
      continue;
    llvm::sort(accesses);

    // 피크에서 접근 중이면 cold가 아니다.
    if (llvm::is_contained(accesses, peakIdx))
      continue;

    // 피크를 포함하는 무접근 구간 [prev, next]
    int64_t prev = -1, next = -1;
    for (int64_t a : accesses) {
      if (a < peakIdx)
        prev = a;
      else {
        next = a;
        break;
      }
    }
    if (prev < 0 || next < 0)
      continue; // 피크 앞 정의가 없거나(불가) 이후 사용이 없음(반환값 등)

    int64_t coldLen = next - prev;
    if (!best || coldLen > best->coldLen ||
        (coldLen == best->coldLen && bytes > best->bytes)) {
      SpillCandidate cand;
      cand.victim = v;
      cand.spillAfter = &*std::next(block->begin(), prev);
      cand.fetchBefore = &*std::next(block->begin(), next);
      cand.coldLen = coldLen;
      cand.bytes = bytes;
      best = cand;
    }
  }
  return best;
}

int64_t expectedPeakAfterSpill(
    func::FuncOp funcOp, Liveness &liveness, const SpillCandidate &cand) {
  int64_t newPeak = 0;
  bool inside = false;
  for (Operation &op : funcOp.getBody().front()) {
    if (&op == cand.fetchBefore)
      inside = false;
    if (isa<ONNXConstantOp>(&op)) {
      if (&op == cand.spillAfter)
        inside = true;
      continue;
    }
    int64_t usage = memoryUsageAtOp(&op, liveness, /*includeConstants=*/false);
    if (inside)
      usage -= cand.bytes;
    newPeak = std::max(newPeak, usage);
    if (&op == cand.spillAfter)
      inside = true;
  }
  return newPeak;
}

int64_t expectedPeakAfterSplit(
    func::FuncOp funcOp, Liveness &liveness, const SplitCandidate &cand) {
  ONNXConvOp pwOp = cand.pwConv;
  ONNXConvOp dwOp = cand.dwConv;
  Operation *pw = pwOp.getOperation();
  Operation *dw = dwOp.getOperation();
  int64_t half = cand.interBytes / 2;
  int64_t newPeak = 0;
  bool inside = false;
  for (Operation &op : funcOp.getBody().front()) {
    if (&op == pw)
      inside = true;
    if (isa<ONNXConstantOp>(&op))
      continue;
    int64_t usage = memoryUsageAtOp(&op, liveness, /*includeConstants=*/false);
    if (inside)
      usage -= half;
    newPeak = std::max(newPeak, usage);
    if (&op == dw)
      inside = false;
  }
  return newPeak;
}

bool applyTensorSpilling(
    SpillCandidate &cand, OpBuilder &builder, int64_t spillId) {
  Value victim = cand.victim;
  Location loc = victim.getLoc();
  auto si64 = [&](int64_t v) { return builder.getI64IntegerAttr(v); };

  // 1. spill: cold range 시작(마지막 접근 직후). victim은 이 지점 이후로
  //    쓰이지 않으므로 spill이 마지막 사용이 되어 버퍼가 여기서 해제된다.
  auto tokenTy =
      RankedTensorType::get({(int64_t)1}, builder.getIntegerType(8));
  builder.setInsertionPointAfter(cand.spillAfter);
  SmallVector<NamedAttribute, 2> spillAttrs{
      builder.getNamedAttr("function_name", builder.getStringAttr("om_spill")),
      builder.getNamedAttr("spill_id", si64(spillId))};
  auto spillOp = builder.create<ONNXCustomOp>(
      loc, TypeRange{tokenTy}, ValueRange{victim}, spillAttrs);
  Value token = spillOp.getResult(0);

  // spill 이후의 victim 사용 지점들 (cold range 구성상 전부 fetchBefore 이후)
  SmallVector<OpOperand *, 4> lateUses;
  for (OpOperand &use : victim.getUses()) {
    if (use.getOwner() == spillOp)
      continue;
    if (!use.getOwner()->isBeforeInBlock(spillOp))
      lateUses.push_back(&use);
  }
  if (lateUses.empty())
    return false; // 있을 수 없는 상황 — 후보 조건상 다음 접근이 존재

  // 2-a. fetch-concat 융합 (논문 Fig. 4(c)): 다음 접근이 2-입력 Concat이고
  //      그것이 victim의 유일한 이후 사용일 때, fetched 텐서를 실체화하지
  //      않고 concat 결과에 바로 쓴다.
  if (lateUses.size() == 1) {
    if (auto concat =
            llvm::dyn_cast<ONNXConcatOp>(lateUses[0]->getOwner())) {
      auto outTy = dyn_cast<RankedTensorType>(concat.getResult().getType());
      auto vicTy = dyn_cast<RankedTensorType>(victim.getType());
      if (concat == cand.fetchBefore && concat.getNumOperands() == 2 &&
          outTy && outTy.hasStaticShape() && vicTy && vicTy.hasStaticShape()) {
        int64_t axis = concat.getAxis();
        if (axis < 0)
          axis += outTy.getRank();
        int64_t fetchPos = (concat.getOperand(0) == victim) ? 0 : 1;
        Value other = concat.getOperand(fetchPos == 0 ? 1 : 0);
        builder.setInsertionPoint(concat);
        SmallVector<NamedAttribute, 4> fusedAttrs{
            builder.getNamedAttr("axis", si64(axis)),
            builder.getNamedAttr("fetch_pos", si64(fetchPos)),
            builder.getNamedAttr("function_name",
                builder.getStringAttr("om_fetch_concat2")),
            builder.getNamedAttr("spill_id", si64(spillId))};
        auto fused = builder.create<ONNXCustomOp>(loc,
            TypeRange{concat.getResult().getType()}, ValueRange{other, token},
            fusedAttrs);
        concat.getResult().replaceAllUsesWith(fused.getResult(0));
        concat.getOperation()->erase();
        return true;
      }
    }
  }

  // 2-b. 일반 fetch: 다음 접근 직전에 되가져오고, 이후 사용 전부를
  //      fetched 값으로 교체한다.
  builder.setInsertionPoint(cand.fetchBefore);
  SmallVector<NamedAttribute, 2> fetchAttrs{
      builder.getNamedAttr("function_name", builder.getStringAttr("om_fetch")),
      builder.getNamedAttr("spill_id", si64(spillId))};
  auto fetchOp = builder.create<ONNXCustomOp>(
      loc, TypeRange{victim.getType()}, ValueRange{token}, fetchAttrs);
  for (OpOperand *use : lateUses)
    use->set(fetchOp.getResult(0));
  return true;
}

} // namespace tinymo
} // namespace onnx_mlir
