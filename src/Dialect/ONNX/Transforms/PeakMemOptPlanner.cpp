//===------ PeakMemOptPlanner.cpp - 분할 계획 수립 ------===//

#include "src/Dialect/ONNX/Transforms/PeakMemOptPlanner.hpp"

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/IR/BuiltinTypes.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace peakmem {

namespace {

// op 하나의 전이 결과를 상태맵과 plan에 반영한다.
// 부분합(Reduced)은 E의 result에서만 허용된다.
bool planApplyStep(Operation *op, Operation *e, const OpSplitStep &step,
    llvm::DenseMap<Value, SplitState> &states, SubgraphPlan &plan,
    unsigned &nSlices) {
  if (step.results.size() != op->getNumResults())
    return false;
  for (auto [idx, st] : llvm::enumerate(step.results)) {
    if (st.isReduced() && op != e) {
      pmoDbg() << "[planner] invalid: partial sums before E at "
               << op->getName() << "\n";
      return false;
    }
    states[op->getResult(idx)] = st;
  }
  nSlices += step.slices.size();
  plan.steps.push_back({op, step});
  return true;
}

// sg.subgraphNodes[startIdx..]를 규칙으로 통과시키며 계획을 채운다.
bool planWalkFrom(Subgraph &sg, unsigned startIdx,
    llvm::DenseMap<Value, SplitState> &states, SubgraphPlan &plan,
    unsigned &nSlices) {
  Operation *e = sg.getE();
  const SplitRuleRegistry &registry = SplitRuleRegistry::instance();
  for (Operation *op : sg.subgraphNodes.getArrayRef().drop_front(startIdx)) {
    const SplitRule *rule = registry.lookup(op);
    if (!rule) {
      pmoDbg() << "[planner] invalid: no rule for " << op->getName() << "\n";
      return false;
    }
    llvm::SmallVector<SplitState, 4> opStates;
    for (Value v : op->getOperands()) {
      auto it = states.find(v);
      opStates.push_back(
          it == states.end() ? SplitState::untouched() : it->second);
    }
    auto step = rule->propagate(op, opStates, plan.nBranches);
    if (failed(step)) {
      pmoDbg() << "[planner] invalid: propagate failed at " << op->getName()
               << "\n";
      return false;
    }
    if (!planApplyStep(op, e, *step, states, plan, nSlices))
      return false;
  }
  return true;
}

// E의 분할 상태로 병합 방법을 확정하고, 피크 추정과 함께 plan을 완성한다.
// runtimeSplitInput: input-split일 때 런타임 Split이 꽂히는 입력(없으면 널).
bool planFinish(Subgraph &sg, Liveness &liveness,
    llvm::DenseMap<Value, SplitState> &states, unsigned nSlices,
    StringRef seedDesc, Value runtimeSplitInput, SubgraphPlan &plan) {
  Operation *s = sg.getS();
  Operation *e = sg.getE();
  const int n = plan.nBranches;
  auto it = states.find(e->getResult(0));
  if (it == states.end())
    return false;
  const SplitState &eState = it->second;
  if (!eState.isPartitioned() && !eState.isReduced())
    return false;

  // 값 하나의 브랜치당 크기: Partitioned는 큰 조각 기준(⌈dim/n⌉/dim),
  // Reduced/외부 값은 풀사이즈. 상수는 호출부에서 걸러진다.
  auto scaledBytes = [&](Value v) -> int64_t {
    // ConcatReuse에서 S(기존 Concat)의 결과는 rewrite 후 실제로 만들어지지
    // 않는다 — 각 브랜치가 Concat의 입력을 직접 소비한다.
    if (plan.seedKind == SubgraphPlan::SeedKind::ConcatReuse &&
        v == s->getResult(0))
      return 0;
    int64_t bytes = getTensorSize(v);
    if (bytes < 0)
      bytes = 0;
    auto sIt = states.find(v);
    if (sIt == states.end() || !sIt->second.isPartitioned())
      return bytes;
    auto rtt = dyn_cast<RankedTensorType>(v.getType());
    if (!rtt)
      return bytes;
    int64_t dim = rtt.getShape()[sIt->second.axis];
    if (dim <= 0)
      return bytes;
    return bytes / dim * ((dim + n - 1) / n);
  };

  // 브랜치 하나를 실행하는 동안의 최대 라이브 합 (분할 후 크기 기준)
  // + 먼저 끝난 브랜치들의 E 출력 보유분 (n-1)개.
  int64_t oldPeak = 0, branchPeak = 0;
  for (Operation *op : sg.subgraphNodes) {
    oldPeak = std::max(oldPeak, getLiveTensorsSize(op, liveness));
    const LivenessBlockInfo *blk = liveness.getLiveness(op->getBlock());
    if (!blk)
      continue;
    llvm::SmallPtrSet<Value, 16> live;
    for (Value v : blk->currentlyLiveValues(op))
      live.insert(v);
    for (Value r : op->getResults())
      live.insert(r);
    int64_t bytes = 0;
    for (Value v : live) {
      if (isFoldedOffline(v))
        continue;
      bytes += scaledBytes(v);
    }
    branchPeak = std::max(branchPeak, bytes);
  }
  int64_t carry = (int64_t)(n - 1) * scaledBytes(e->getResult(0));
  int64_t newPeak = branchPeak + carry;
  // input-split이면 Split 시점에 원본과 조각들이 잠시 공존한다: 2x|X|.
  if (runtimeSplitInput) {
    int64_t x = getTensorSize(runtimeSplitInput);
    if (x > 0)
      newPeak = std::max(newPeak, 2 * x);
  }

  plan.mergeIsAdd = eState.isReduced();
  plan.mergeAxis = eState.isPartitioned() ? eState.axis : -1;
  plan.oldPeak = oldPeak;
  plan.newPeak = newPeak;

  pmoDbg() << "[planner] plan: split=" << seedDesc << " merge=";
  if (eState.isPartitioned())
    pmoDbg() << "Concat(axis=" << eState.axis << ")";
  else
    pmoDbg() << "Add";
  pmoDbg() << " branches=" << n << " sliceReqs=" << nSlices << "\n";
  pmoDbg() << "[planner] est: regionPeak " << oldPeak << "B -> " << newPeak
           << "B (reduction " << (oldPeak - newPeak) << "B)\n";
  return true;
}

} // namespace

std::optional<SubgraphPlan> computePlan(Subgraph &sg, Liveness &liveness) {
  Operation *s = sg.getS();
  Operation *e = sg.getE();
  if (!s || !e || s == e || e->getNumResults() != 1) {
    pmoDbg() << "[planner] skip: malformed subgraph\n";
    return std::nullopt;
  }

  SubgraphPlan plan;
  const SplitRule *sRule = SplitRuleRegistry::instance().lookup(s);
  if (!sRule) {
    pmoDbg() << "[planner] invalid: no rule for S " << s->getName() << "\n";
    return std::nullopt;
  }

  // S는 weight-split만 허용한다(2026-07-21 사용자 결정): 시드는 S의 상수
  // operand를 오프라인으로 자르는 것뿐이고, 후속 연산들은 전파로 쪼개진
  // 입력을 받는다. S의 activation에 런타임 Split을 꽂는 input-split 시딩은
  // 제거됨.
  if (auto seed = sRule->planWeightSplit(s, plan.nBranches); succeeded(seed)) {
    llvm::DenseMap<Value, SplitState> states;
    unsigned nSlices = 0;
    if (planApplyStep(s, e, *seed, states, plan, nSlices) &&
        planWalkFrom(sg, 1, states, plan, nSlices) &&
        planFinish(
            sg, liveness, states, nSlices, "weight-split", Value(), plan))
      return plan;
  }
  pmoDbg() << "[planner] invalid: no viable split plan\n";
  return std::nullopt;
}

std::optional<SubgraphPlan> computeConcatReusePlan(
    Subgraph &sg, Liveness &liveness) {
  Operation *s = sg.getS();
  Operation *e = sg.getE();
  if (!s || !e || s == e || e->getNumResults() != 1)
    return std::nullopt;
  auto concatOp = dyn_cast<ONNXConcatOp>(s);
  if (!concatOp || s->getNumOperands() < 2 || s->getNumResults() != 1)
    return std::nullopt;
  auto outTy = dyn_cast<RankedTensorType>(s->getResult(0).getType());
  if (!outTy || !outTy.hasStaticShape())
    return std::nullopt;
  int64_t axis = concatOp.getAxis();
  if (axis < 0)
    axis += outTy.getRank();

  int64_t width = -1;
  for (Value in : s->getOperands()) {
    auto ty = dyn_cast<RankedTensorType>(in.getType());
    if (!ty || !ty.hasStaticShape())
      return std::nullopt;
    int64_t w = ty.getShape()[axis];
    if (width == -1)
      width = w;
    else if (w != width) {
      pmoDbg() << "[planner] invalid: concat inputs are uneven along the "
                  "merge axis\n";
      return std::nullopt;
    }
  }

  SubgraphPlan plan;
  plan.seedKind = SubgraphPlan::SeedKind::ConcatReuse;
  plan.nBranches = (int)s->getNumOperands();
  llvm::DenseMap<Value, SplitState> states;
  states[s->getResult(0)] = SplitState::partitioned(axis);
  unsigned nSlices = 0;
  if (!planWalkFrom(sg, 1, states, plan, nSlices))
    return std::nullopt;
  if (!planFinish(
          sg, liveness, states, nSlices, "concat-reuse", Value(), plan))
    return std::nullopt;
  return plan;
}

} // namespace peakmem
} // namespace onnx_mlir
