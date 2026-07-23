//===------ PeakMemOptPlanner.hpp - 분할 계획/집행 ------===//
//
// Plan 단계: 매칭된 서브그래프에 대해 SplitRule로 분할 상태를 전파해
// "실행 가능한 계획(SubgraphPlan)"을 세우고 피크 감소량을 추정한다.
// Execute 단계: 계획을 IR에 집행한다. 둘 다 op-agnostic — op 특성은 전부
// PeakMemOptRules의 규칙에서 온다.
//
//===----------------------------------------------------------------===//

#pragma once

#include <optional>

#include "src/Dialect/ONNX/Transforms/PeakMemOptRules.hpp"
#include "src/Dialect/ONNX/Transforms/PeakMemOptUtils.hpp"

namespace onnx_mlir {
namespace peakmem {

/// 서브그래프 하나에 대한 실행 가능한 분할 계획. steps는 토폴로지 순서다.
/// seedKind에 따라:
///  - WeightSplit: steps[0] == S (planWeightSplit의 답)
///  - ConcatReuse: S(기존 Concat)는 클론하지 않고 지우므로 steps에 없다.
///                 브랜치 i의 입력은 S의 i번째 operand.
struct SubgraphPlan {
  enum class SeedKind { WeightSplit, ConcatReuse };
  SeedKind seedKind = SeedKind::WeightSplit;
  int nBranches = kNumBranches;
  llvm::SmallVector<std::pair<mlir::Operation *, OpSplitStep>, 8> steps;
  bool mergeIsAdd = false;
  int64_t mergeAxis = -1;           // Concat 병합일 때의 축
  int64_t oldPeak = 0, newPeak = 0; // 전역 추정치 (바이트, 참고용)
  /// benefit analysis (논문 §3.3) 결과: 영역-국소 모델의 기대 피크 감소
  /// = M_p − max{조건별 LHS}. 조건 하나라도 위배되면 후보 자체가 기각되므로
  /// 유효 plan에서는 항상 > 0.
  int64_t benefitReduction = 0;
};

/// expand/shrink · fork/join 서브그래프의 계획: S의 weight-split만 시도한다
/// (S는 weight-split만 허용 — 2026-07-21 결정). IR은 변경하지 않는다.
std::optional<SubgraphPlan> computePlan(Subgraph &sg, mlir::Liveness &liveness);

/// concat-shrink(merge-shrink) 서브그래프의 계획: 기존 Concat의 입력들을
/// 브랜치 입력으로 재사용한다(런타임 Split 불필요). 브랜치 수는 Concat의
/// 입력 개수. 반대편 상수 슬라이스가 균등 분할로 만들어지므로 모든 Concat
/// 입력이 병합 축에서 같은 크기일 때만 계획을 세운다.
std::optional<SubgraphPlan> computeConcatReusePlan(
    Subgraph &sg, mlir::Liveness &liveness);

/// 계획을 IR에 집행한다. 새 op는 전부 원본 S 바로 앞에 순서대로 삽입되고
/// (슬라이스 → 브랜치들 → 병합), 마지막에 원본 서브그래프를 지운다.
void executePlan(Subgraph &sg, const SubgraphPlan &plan);

} // namespace peakmem
} // namespace onnx_mlir
