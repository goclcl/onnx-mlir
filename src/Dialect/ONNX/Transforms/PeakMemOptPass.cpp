//===------ PeakMemOptPass.cpp - 피크 메모리 최적화 드라이버 ------===//
//
// 파이프라인: Match(3 matcher) → Plan(규칙 기반) → Select(추정 감소량 최대)
// → Execute(단일 엔진). 이 파일은 드라이버만 담당한다:
//  - rewrite 1회마다 liveness를 새로 계산해 전역 피크를 찾고,
//  - 피크 op마다 세 패턴의 후보 전부에 계획을 세워 최선을 실행하며,
//  - 첫 분석의 피크 값(wave)이 해소되면 종료한다.
//
// 디버그 출력: --debug-only=peak-mem-opt (디버그 빌드, stderr).
//
//===----------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/PeakMemOptMatchers.hpp"
#include "src/Dialect/ONNX/Transforms/PeakMemOptPlanner.hpp"
#include "src/Dialect/ONNX/Transforms/PeakMemOptUtils.hpp"

using namespace mlir;
using namespace onnx_mlir::peakmem;

namespace {

struct PeakMemOptPass
    : public PassWrapper<PeakMemOptPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PeakMemOptPass)

  StringRef getArgument() const override { return "onnx-peak-mem-opt"; }
  StringRef getDescription() const override {
    return "Performs graph transformations to lower the peak memory usage.";
  }

  PeakMemOptPass() = default;
  PeakMemOptPass(const PeakMemOptPass &pass)
      : PassWrapper<PeakMemOptPass, OperationPass<func::FuncOp>>() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    // 무한 루프 방지용 안전 상한 (정상 경로는 피크 변화/후보 소진으로 종료).
    constexpr int kMaxRewrites = 32;

    int rewrites = 0;
    int64_t targetPeak = -1;
    for (int iter = 0; iter < kMaxRewrites; ++iter) {
      if (!tryRewriteOnce(funcOp, targetPeak, iter))
        break;
      ++rewrites;
    }
    pmoDbg() << "[driver] done: " << rewrites << " rewrite(s)\n";
  }

  // 분석 → 매칭 → 계획 → 선택 → 실행 사이클 1회. rewrite가 일어나면 true.
  //
  // 매 호출마다 liveness를 새로 계산한다(rewrite 후의 재사용은 stale).
  // targetPeak은 첫 분석의 전역 피크 값으로, 이 실행(wave)이 낮추려는
  // 대상이다: 같은 값으로 묶인 피크 영역들을 모두 처리해 피크가 실제로
  // 변하면 종료한다. 낮아진 피크를 계속 다시 쪼개는 것은 의도적으로 하지
  // 않는다(분할 깊이가 무한히 깊어지는 것을 막는 1-wave 계약).
  bool tryRewriteOnce(func::FuncOp funcOp, int64_t &targetPeak, int iter) {
    Liveness liveness(funcOp);

    /* ===============Peak 탐색=============== */
    llvm::SmallVector<Operation *, 8> peakOps;
    int64_t peakBytes = -1;
    funcOp.walk([&](Operation *op) {
      // onnx.Constant와 func.func op는 스킵
      if (isa<ONNXConstantOp>(op) || isa<func::FuncOp>(op))
        return;

      int64_t liveTensorsSize = getLiveTensorsSize(op, liveness);

      if (liveTensorsSize > peakBytes) {
        peakOps.clear();
        peakBytes = liveTensorsSize;
        peakOps.push_back(op);
      } else if (liveTensorsSize == peakBytes) {
        peakOps.push_back(op);
      }
    });

    if (targetPeak == -1)
      targetPeak = peakBytes;
    if (peakBytes != targetPeak) {
      pmoDbg() << "[driver] peak changed " << targetPeak << "B -> " << peakBytes
               << "B; done\n";
      return false;
    }

    pmoDbg() << "\n[driver] iteration " << iter << ": peak " << peakBytes
             << "B at " << peakOps.size() << " op(s)\n";

    for (Operation *peakOp : peakOps) {
      /* ===============Match=============== */
      Subgraph expandShrinkSubgraph = matchExpandShrinkPattern(peakOp);
      Subgraph forkJoinSubgraph = matchForkJoinPattern(peakOp);
      Subgraph concatShrinkSubgraph = matchMergeShrinkPattern(peakOp);

      /* ===============Plan & Select & Rewrite=============== */
      // 매칭된 후보 전부에 대해 계획을 세우고, 가장 작은 서브그래프(노드 수
      // 최소)에서 쪼개는 후보를 고른다 — 교란과 클로닝이 가장 적은 지점을
      // 선호한다. 동률이면 추정 피크 감소량이 큰 쪽. 세 패턴은 같은
      // 계획/엔진을 쓰며, expand/shrink(단일 체인)와 fork/join(DAG)은
      // 정의상 같은 영역을 동시에 주장할 수 없다.
      struct Candidate {
        Subgraph *sg;
        SubgraphPlan plan;
        const char *kind;
      };
      llvm::SmallVector<Candidate, 3> candidates;
      if (expandShrinkSubgraph.getS())
        if (auto p = computePlan(expandShrinkSubgraph, liveness))
          candidates.push_back({&expandShrinkSubgraph, *p, "expand/shrink"});
      if (forkJoinSubgraph.getS())
        if (auto p = computePlan(forkJoinSubgraph, liveness))
          candidates.push_back({&forkJoinSubgraph, *p, "fork/join"});
      if (concatShrinkSubgraph.getS())
        if (auto p = computeConcatReusePlan(concatShrinkSubgraph, liveness))
          candidates.push_back({&concatShrinkSubgraph, *p, "concat-shrink"});

      if (candidates.empty()) {
        pmoDbg() << "[select] no viable plan at this peak op\n";
        continue;
      }

      Candidate *best = &candidates[0];
      for (Candidate &c : llvm::drop_begin(candidates)) {
        size_t nBest = best->sg->subgraphNodes.size();
        size_t nC = c.sg->subgraphNodes.size();
        int64_t rBest = best->plan.oldPeak - best->plan.newPeak;
        int64_t rC = c.plan.oldPeak - c.plan.newPeak;
        if (nC < nBest || (nC == nBest && rC > rBest))
          best = &c;
      }
      pmoDbg() << "[select] " << candidates.size() << " candidate(s) -> "
               << best->kind << " (" << best->sg->subgraphNodes.size()
               << " nodes, est. reduction "
               << (best->plan.oldPeak - best->plan.newPeak) << "B)\n";

      executePlan(*best->sg, best->plan);
      return true;
    }
    return false;
  }
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createPeakMemOptPass() {
  return std::make_unique<PeakMemOptPass>();
}
} // namespace onnx_mlir
