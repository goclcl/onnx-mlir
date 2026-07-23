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
bool planFinish(Subgraph &sg, Liveness &liveness,
    llvm::DenseMap<Value, SplitState> &states, unsigned nSlices,
    StringRef seedDesc, SubgraphPlan &plan) {
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


//===--------------------- Benefit analysis (논문 §3.3) ---------------------===//
//
// 영역-국소 모델로 분할의 이득을 판정한다. 순차 브랜치 스케줄을 가정하고,
//  fork/join·expand/shrink (N=2, 브랜치 a→b 순):
//   (i)   M_p^(a) + M_i < M_p          (b가 남아있는 동안 입력이 살아있음)
//   (ii)  M_o^(a) + M_p^(b) < M_p      (b 실행 중 a의 부분 출력이 살아있음)
//   (iii) ΣM_o^(k) + M_o < M_p        (병합 시점: 부분 출력들 + 최종 출력)
//  merge/shrink (K=concat 입력 수, 브랜치 k 실행 중):
//   (i)   Σ_{j<k}M_o^(j) + M_p^(k) + Σ_{j>k}M_i^(j) < M_p   (모든 k)
//   (ii)  Σ_k M_o^(k) + M_o < M_p
// 모든 조건 성립 시에만 분할하며, 기대 감소 = M_p − max{각 조건 LHS}.
//
// 크기 계정은 영역 내부 텐서만으로 하며(외부 상주 텐서는 양변 공통이라
// 소거), M_i는 원본에서 S 시점에만 산 것으로, 브랜치 모델에서는 조건이
// 명시적으로 더한다.

// 영역 내부 값들의 [정의, 마지막 사용] 구간 기반 liveness로, 각 op 시점의
// 크기 합 최댓값을 계산한다. scale(v)로 값별 크기를 브랜치 배율로 조정한다.
int64_t regionPeakWith(Subgraph &sg,
    llvm::function_ref<int64_t(mlir::Value)> scale) {
  llvm::DenseMap<Operation *, unsigned> pos;
  for (auto [i, op] : llvm::enumerate(sg.subgraphNodes.getArrayRef()))
    pos[op] = i;
  unsigned n = sg.subgraphNodes.size();
  // 각 op 시점에 라이브인 내부 값: def(v) <= t <= lastUse(v)
  llvm::SmallVector<int64_t, 16> liveAt(n, 0);
  for (Operation *op : sg.subgraphNodes) {
    for (Value r : op->getResults()) {
      if (isFoldedOffline(r) || isa<NoneType>(r.getType()))
        continue;
      unsigned d = pos[op], lu = d;
      for (Operation *user : r.getUsers()) {
        auto it = pos.find(user);
        if (it != pos.end())
          lu = std::max(lu, it->second);
      }
      int64_t sz = scale(r);
      for (unsigned t = d; t <= lu; ++t)
        liveAt[t] += sz;
    }
  }
  int64_t peak = 0;
  for (int64_t x : liveAt)
    peak = std::max(peak, x);
  return peak;
}

// 브랜치 i에서의 값 크기: Partitioned면 그 브랜치의 조각 크기(불균등은
// 앞 브랜치가 나머지), Reduced/Untouched는 풀사이즈.
int64_t branchScaledSize(Value v, const llvm::DenseMap<Value, SplitState> &st,
    int branchIdx, int nBranches) {
  int64_t bytes = getTensorSize(v);
  if (bytes <= 0)
    return 0;
  auto it = st.find(v);
  if (it == st.end() || !it->second.isPartitioned())
    return bytes;
  auto rtt = dyn_cast<RankedTensorType>(v.getType());
  if (!rtt)
    return bytes;
  int64_t dim = rtt.getShape()[it->second.axis];
  if (dim <= 0)
    return bytes;
  int64_t base = dim / nBranches, rem = dim % nBranches;
  return bytes / dim * (base + (branchIdx < rem ? 1 : 0));
}

// fork/join·expand/shrink(WeightSplit 시드)의 benefit 판정.
// 성립 시 plan.benefitReduction을 채우고 true.
bool benefitWeightSplit(Subgraph &sg,
    const llvm::DenseMap<Value, SplitState> &states, SubgraphPlan &plan) {
  Operation *s = sg.getS();
  Operation *e = sg.getE();
  const int n = plan.nBranches;

  // M_i: S의 활성 입력(상수 제외) 합. M_o: E 출력 풀사이즈.
  int64_t Mi = 0;
  for (Value v : s->getOperands())
    if (!isFoldedOffline(v) && !isa<NoneType>(v.getType()))
      Mi += std::max<int64_t>(getTensorSize(v), 0);
  int64_t Mo = std::max<int64_t>(getTensorSize(e->getResult(0)), 0);

  // M_p: 원본 영역 피크(내부 텐서 + S 시점의 M_i).
  int64_t Mp = regionPeakWith(sg, [](Value v) {
    return std::max<int64_t>(getTensorSize(v), 0);
  });
  // 원본에서 M_i는 S 시점에만 살아있다: liveAt[S] = S 결과뿐이므로
  // max(내부피크, M_i + |S결과|)가 정확하다.
  Mp = std::max(
      Mp, Mi + std::max<int64_t>(getTensorSize(s->getResult(0)), 0));

  // 브랜치별 피크 M_p^(k)와 출력 M_o^(k).
  llvm::SmallVector<int64_t, 2> Mpk(n), Mok(n);
  for (int k = 0; k < n; ++k) {
    Mpk[k] = regionPeakWith(sg, [&](Value v) {
      return branchScaledSize(v, states, k, n);
    });
    Mok[k] = branchScaledSize(e->getResult(0), states, k, n);
  }

  // 조건 LHS들 (N=2 일반형: 브랜치 k 실행 중 = 앞 브랜치 출력들 + 자기
  // 피크 + (마지막이 아니면) 입력 M_i), 마지막으로 병합 시점.
  int64_t worst = 0;
  bool ok = true;
  int64_t acc = 0; // Σ_{j<k} M_o^(j)
  for (int k = 0; k < n; ++k) {
    int64_t lhs = acc + Mpk[k] + (k + 1 < n ? Mi : 0);
    pmoDbg() << "[benefit] branch " << k << ": " << lhs << (lhs < Mp ? " < " : " >= ")
             << Mp << "\n";
    ok &= lhs < Mp;
    worst = std::max(worst, lhs);
    acc += Mok[k];
  }
  int64_t mergeLhs = acc + Mo; // 병합 시점: ΣM_o^(k) + M_o
  pmoDbg() << "[benefit] merge: " << mergeLhs << (mergeLhs < Mp ? " < " : " >= ")
           << Mp << "\n";
  ok &= mergeLhs < Mp;
  worst = std::max(worst, mergeLhs);

  if (!ok) {
    pmoDbg() << "[planner] invalid: not beneficial (benefit analysis)\n";
    return false;
  }
  plan.benefitReduction = Mp - worst;
  pmoDbg() << "[benefit] expected reduction " << plan.benefitReduction
           << "B (Mp " << Mp << "B)\n";
  return true;
}

// merge/shrink(ConcatReuse)의 benefit 판정. 브랜치 수 = Concat 입력 수.
bool benefitConcatReuse(Subgraph &sg,
    const llvm::DenseMap<Value, SplitState> &states, SubgraphPlan &plan) {
  Operation *s = sg.getS(); // 기존 Concat
  Operation *e = sg.getE();
  const int n = plan.nBranches;

  llvm::SmallVector<int64_t, 4> Mik(n);
  for (int k = 0; k < n; ++k)
    Mik[k] = std::max<int64_t>(getTensorSize(s->getOperand(k)), 0);
  int64_t Mo = std::max<int64_t>(getTensorSize(e->getResult(0)), 0);

  // M_p: 원본 영역 피크. Concat 시점엔 입력 전부 + 출력이 공존한다.
  int64_t Mp = regionPeakWith(sg, [](Value v) {
    return std::max<int64_t>(getTensorSize(v), 0);
  });
  int64_t sumMi = 0;
  for (int64_t x : Mik)
    sumMi += x;
  Mp = std::max(Mp, sumMi + std::max<int64_t>(
                        getTensorSize(s->getResult(0)), 0));

  // 브랜치 k: 클론 영역(S 제외)의 피크·출력. Partitioned 값은 그 브랜치
  // 조각, Reduced는 풀사이즈(M_o^(k) = M_o).
  llvm::SmallVector<int64_t, 4> Mpk(n), Mok(n);
  for (int k = 0; k < n; ++k) {
    Mpk[k] = regionPeakWith(sg, [&](Value v) {
      if (v.getDefiningOp() == s)
        return (int64_t)0; // Concat 결과는 생성되지 않음 (입력 직접 소비)
      return branchScaledSize(v, states, k, n);
    });
    // 브랜치 입력 자체(M_i^(k))는 조건식이 명시적으로 다루므로 피크에서
    // 제외돼 있고(위에서 S 결과=0), 여기에 브랜치 실행 중 자기 입력을 더함.
    Mpk[k] += Mik[k];
    Mok[k] = branchScaledSize(e->getResult(0), states, k, n);
  }

  bool ok = true;
  int64_t worst = 0, accOut = 0;
  for (int k = 0; k < n; ++k) {
    int64_t after = 0;
    for (int j = k + 1; j < n; ++j)
      after += Mik[j];
    int64_t lhs = accOut + Mpk[k] + after;
    pmoDbg() << "[benefit] branch " << k << ": " << lhs
             << (lhs < Mp ? " < " : " >= ") << Mp << "\n";
    ok &= lhs < Mp;
    worst = std::max(worst, lhs);
    accOut += Mok[k];
  }
  int64_t mergeLhs = accOut + Mo;
  pmoDbg() << "[benefit] merge: " << mergeLhs
           << (mergeLhs < Mp ? " < " : " >= ") << Mp << "\n";
  ok &= mergeLhs < Mp;
  worst = std::max(worst, mergeLhs);

  if (!ok) {
    pmoDbg() << "[planner] invalid: not beneficial (benefit analysis)\n";
    return false;
  }
  plan.benefitReduction = Mp - worst;
  pmoDbg() << "[benefit] expected reduction " << plan.benefitReduction
           << "B (Mp " << Mp << "B)\n";
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
        planFinish(sg, liveness, states, nSlices, "weight-split", plan) &&
        benefitWeightSplit(sg, states, plan))
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
  if (!planFinish(sg, liveness, states, nSlices, "concat-reuse", plan))
    return std::nullopt;
  if (!benefitConcatReuse(sg, states, plan))
    return std::nullopt;
  return plan;
}

} // namespace peakmem
} // namespace onnx_mlir
