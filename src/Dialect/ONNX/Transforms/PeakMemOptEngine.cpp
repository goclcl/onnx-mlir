//===------ PeakMemOptEngine.cpp - 분할 계획 집행 엔진 ------===//
//
// 계획(SubgraphPlan)을 IR에 집행한다. 규칙은 선언(SliceReq/patchClone)만
// 하고, 모든 IR 생성/변경은 여기서 일어난다. 엔진의 계약:
//  - 슬라이스(Split 등)와 hoist된 상수는 그것을 사용할 모든 클론을
//    지배하는 위치(원본 S 앞)에 생성한다.
//  - 클론의 결과 타입은 분할 상태로부터 직접 계산해 설정한다.
//
//===----------------------------------------------------------------===//

#include "src/Dialect/ONNX/Transforms/PeakMemOptPlanner.hpp"

#include <functional>

#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace peakmem {

void executePlan(Subgraph &sg, const SubgraphPlan &plan) {
  const int n = plan.nBranches;
  Operation *s = sg.getS();
  Operation *e = sg.getE();
  Location loc = s->getLoc();
  OpBuilder builder(s);

  // v의 정의가 삽입 영역(원본 S 앞)을 지배하지 않으면 — 원본 그래프에서
  // S 뒤에 정의된 상수류 — insertBefore 앞으로 복제해 지배시킨다.
  // 상수/NoValue뿐 아니라 상수의 Split 조각(앞선 재작성이 만든 가중치
  // 슬라이스)도 오퍼랜드까지 재귀적으로 hoist한다.
  llvm::DenseMap<Value, Value> hoisted;
  std::function<Value(Value, Operation *)> ensureDominates =
      [&](Value v, Operation *insertBefore) -> Value {
    Operation *def = v.getDefiningOp();
    if (!def || def->getBlock() != s->getBlock() || def->isBeforeInBlock(s))
      return v;
    auto it = hoisted.find(v);
    if (it != hoisted.end())
      return it->second;
    if (!isOfflineProducer(def)) {
      pmoDbg() << "[engine] warning: cannot hoist non-constant "
               << def->getName() << "\n";
      return v;
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(insertBefore);
    IRMapping map;
    for (Value opnd : def->getOperands())
      map.map(opnd, ensureDominates(opnd, insertBefore));
    Operation *copy = builder.clone(*def, map);
    for (auto [ri, r] : llvm::enumerate(def->getResults()))
      hoisted[r] = copy->getResult(ri);
    return hoisted[v];
  };

  // Partition 슬라이스 materialize: v의 axis를 n조각으로.
  // 불균등이면 앞쪽 조각에 나머지를 배분한다.
  auto makeSlices = [&](Value v, int64_t axis) -> SmallVector<Value, 2> {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(s);
    Value src = ensureDominates(v, s);
    auto ty = cast<RankedTensorType>(src.getType());
    ArrayRef<int64_t> shape = ty.getShape();
    int64_t dim = shape[axis];
    int64_t base = dim / n, rem = dim % n;
    SmallVector<int64_t, 2> sizes;
    SmallVector<Type, 2> outTypes;
    for (int64_t i = 0; i < n; ++i) {
      SmallVector<int64_t, 4> outShape(shape.begin(), shape.end());
      outShape[axis] = base + (i < rem ? 1 : 0);
      sizes.push_back(outShape[axis]);
      outTypes.push_back(RankedTensorType::get(outShape, ty.getElementType()));
    }
    Value splitSizes = builder.create<ONNXNoneOp>(loc).getResult();
    if (rem != 0) {
      auto sizesTy = RankedTensorType::get({(int64_t)n}, builder.getI64Type());
      splitSizes =
          builder
              .create<ONNXConstantOp>(loc, Attribute(),
                  DenseElementsAttr::get(sizesTy, ArrayRef<int64_t>(sizes)))
              .getResult();
    }
    auto si64 = builder.getIntegerType(64, /*isSigned=*/true);
    auto splitOp = builder.create<ONNXSplitOp>(loc, TypeRange(outTypes), src,
        splitSizes, builder.getIntegerAttr(si64, axis),
        builder.getIntegerAttr(si64, (int64_t)n));
    return SmallVector<Value, 2>(
        splitOp.getResults().begin(), splitOp.getResults().end());
  };

  // 1. 모든 Partition 요구를 먼저 materialize (조각들은 브랜치가 공유)
  SmallVector<SmallVector<SmallVector<Value, 2>, 2>, 8> partSlices(
      plan.steps.size());
  for (auto [si, entry] : llvm::enumerate(plan.steps)) {
    for (const SliceReq &req : entry.second.slices) {
      if (req.kind == SliceReq::Kind::Partition)
        partSlices[si].push_back(
            makeSlices(entry.first->getOperand(req.operandIdx), req.axis));
      else
        partSlices[si].push_back({}); // ApplyOnce: 인덱스 자리 맞춤용
    }
  }
  SmallVector<Value, 2> seedSlices;
  if (plan.seedKind == SubgraphPlan::SeedKind::InputSplit)
    seedSlices = makeSlices(
        s->getOperand(plan.inputSeed.operandIdx), plan.inputSeed.inAxis);
  Value noneVal; // ApplyOnce의 비-첫 브랜치용 (지연 생성)

  // 2. 브랜치 생성
  builder.setInsertionPoint(s);
  SmallVector<Value, 2> branchOuts;
  for (int i = 0; i < n; ++i) {
    IRMapping mapping;
    if (plan.seedKind == SubgraphPlan::SeedKind::InputSplit)
      mapping.map(s->getOperand(plan.inputSeed.operandIdx), seedSlices[i]);
    else if (plan.seedKind == SubgraphPlan::SeedKind::ConcatReuse)
      mapping.map(s->getResult(0), s->getOperand(i));
    for (auto [si, entry] : llvm::enumerate(plan.steps)) {
      Operation *op = entry.first;
      const OpSplitStep &step = entry.second;
      Operation *cloned = builder.clone(*op, mapping);
      for (auto [ri, req] : llvm::enumerate(step.slices)) {
        if (req.kind == SliceReq::Kind::Partition) {
          cloned->setOperand(req.operandIdx, partSlices[si][ri][i]);
        } else if (i != 0) { // ApplyOnce: 브랜치 0만 원본을 유지
          if (!noneVal) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(cloned);
            noneVal = builder.create<ONNXNoneOp>(loc).getResult();
          }
          cloned->setOperand(req.operandIdx, noneVal);
        }
      }
      // 슬라이스 대상이 아닌 operand 중 원본에서 S 뒤에 정의된 상수
      // (Reshape shape, Clip min/max 등)는 클론 앞으로 hoist.
      for (unsigned oi = 0; oi < cloned->getNumOperands(); ++oi)
        cloned->setOperand(oi, ensureDominates(cloned->getOperand(oi), cloned));
      if (const SplitRule *rule = SplitRuleRegistry::instance().lookup(op))
        rule->patchClone(builder, cloned, i, n, step);
      // 결과 타입을 분할 상태로부터 직접 계산해 설정한다: Partitioned는
      // 이 브랜치의 조각 크기, Reduced/Untouched는 원본 크기. 다음
      // iteration의 liveness 재분석이 정확한 크기를 보게 하기 위함이며,
      // 이후의 shape inference와도 일치한다.
      for (auto [ri2, r] : llvm::enumerate(cloned->getResults())) {
        if (isa<NoneType>(r.getType()))
          continue; // 사용되지 않는 선택적 결과는 그대로 둔다
        const SplitState &st = step.results[ri2];
        auto origTy = dyn_cast<RankedTensorType>(op->getResult(ri2).getType());
        if (!origTy || !origTy.hasStaticShape()) {
          r.setType(UnrankedTensorType::get(
              cast<TensorType>(r.getType()).getElementType()));
          continue;
        }
        if (st.isPartitioned()) {
          SmallVector<int64_t, 4> shape(
              origTy.getShape().begin(), origTy.getShape().end());
          int64_t dim = shape[st.axis];
          int64_t base = dim / n, rem = dim % n;
          shape[st.axis] = base + (i < rem ? 1 : 0);
          r.setType(RankedTensorType::get(shape, origTy.getElementType()));
        } else {
          r.setType(origTy); // Reduced/Untouched: 원본과 동일한 모양
        }
      }
    }
    branchOuts.push_back(mapping.lookup(e->getResult(0)));
  }

  // 3. 병합 (Add는 왼쪽부터 접는 체인으로 n-way 지원).
  // 병합 결과는 원본 E와 같은 모양이므로 타입을 그대로 사용한다.
  Type mergedTy = e->getResult(0).getType();
  Value merged;
  if (plan.mergeIsAdd) {
    merged = branchOuts[0];
    for (int i = 1; i < n; ++i)
      merged = builder.create<ONNXAddOp>(loc, mergedTy, merged, branchOuts[i])
                   .getResult();
  } else {
    auto si64 = builder.getIntegerType(64, /*isSigned=*/true);
    merged = builder
                 .create<ONNXConcatOp>(loc, mergedTy, ValueRange(branchOuts),
                     builder.getIntegerAttr(si64, plan.mergeAxis))
                 .getResult();
  }
  e->getResult(0).replaceAllUsesWith(merged);

  // 4. 원본 서브그래프 제거 (역토포 순)
  for (auto it = sg.subgraphNodes.rbegin(); it != sg.subgraphNodes.rend(); ++it)
    (*it)->erase();
  sg.subgraphNodes.clear();

  pmoDbg() << "[engine] rewrite done: " << plan.steps.size()
           << " ops per branch, " << n << " branches\n";
}

} // namespace peakmem
} // namespace onnx_mlir
