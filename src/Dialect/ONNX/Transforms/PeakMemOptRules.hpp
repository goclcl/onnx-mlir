//===------ PeakMemOptRules.hpp - op별 분할 규칙 인터페이스 ------===//
//
// PeakMemOptPass의 Plan/Rewrite 단계가 사용하는 op-agnostic 인터페이스.
// 엔진(피크 탐색/패턴 매칭/축 전파/클론/병합)은 구체적인 op 이름을 모르며,
// op 특성은 전부 SplitRule 구현체로 캡슐화한다.
// 새 op 지원 = 규칙 1개 등록. 엔진 수정 0줄.
//
// 용어: 분할 대상은 S(시작 op)부터 E(끝 op)까지로 닫히는 서브그래프이며,
// S에서 만들어진 N개의 브랜치는 E의 result 직후에 병합된다.
//
// 설계 결정:
//  - N(브랜치 수) = 2 고정. merge-shrink 패턴만 예외적으로 기존 Concat의
//    입력 개수를 그대로 사용한다.
//  - Reduced(부분합)는 E의 result에서만 허용. 부분합은 브랜치당 풀사이즈라
//    중간 전파는 live tensor 합을 줄이지 못해 피크 감소 목적에 반한다.
//  - 시드는 S의 weight-split뿐이다(2026-07-21 결정): S의 "상수" operand
//    (weight/bias)를 잘라 result를 파티션시킨다. 상수의 Split은 상수폴딩으로
//    컴파일 타임에 접혀 런타임 오버헤드 0. 후속 연산들은 전파로 쪼개진
//    입력을 받는다. S의 activation에 런타임 onnx.Split을 꽂는 input-split
//    시딩과 S의 출력을 자르는 방식은 쓰지 않는다 — 전자는 풀사이즈 copy
//    비용, 후자는 잘리기 전의 풀사이즈 출력이 그대로 생성되어 피크가
//    줄지 않는다.
//  - batch(축 0)는 분할 후보에서 원천 제외.
//
//===----------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace onnx_mlir {
namespace peakmem {

/// 브랜치 수. 기존 Concat의 입력들을 브랜치로 재사용하는 패턴(merge-shrink)만
/// 예외로 그 입력 개수를 쓰고, 나머지는 모두 이 값.
constexpr int kNumBranches = 2;

/// 분할이 SSA 값 하나에 미치는 영향을 나타내는 "분할 상태". Plan 단계의
/// 일은 서브그래프 내부의 모든 값에 대해 이 상태를 계산해 두는 것이 전부다.
struct SplitState {
  enum class Kind {
    Untouched,   // 분할과 무관 (broadcast 상수, 서브그래프 외부 값)
    Partitioned, // axis 차원이 N조각으로 파티션됨 (브랜치는 1/N만 봄)
    Reduced      // 부분합 (조각을 모두 더해야 원래 값; E의 result에서만 허용)
  };
  Kind kind = Kind::Untouched;
  int64_t axis = -1; // Partitioned일 때만 유효

  static SplitState untouched() { return {Kind::Untouched, -1}; }
  static SplitState partitioned(int64_t d) { return {Kind::Partitioned, d}; }
  static SplitState reduced() { return {Kind::Reduced, -1}; }

  bool isUntouched() const { return kind == Kind::Untouched; }
  bool isPartitioned() const { return kind == Kind::Partitioned; }
  bool isReduced() const { return kind == Kind::Reduced; }
};

/// 규칙이 엔진에게 내는 "요구서": 이 operand를 이 축으로 잘라 달라.
/// 규칙은 선언만 하고 IR 생성은 엔진이 한다. 엔진의 계약: 슬라이스(Split 등)는
/// 원본 operand의 정의 위치가 아니라, 그것을 사용할 모든 클론을 지배하는
/// 위치에 생성해야 한다.
struct SliceReq {
  unsigned operandIdx;
  int64_t axis;
  enum class Kind {
    Partition, // 브랜치 i가 i번째 조각을 받음
    ApplyOnce  // 부분합 병합 뒤 한 번만 적용되어야 하는 가산항(bias 등).
               // 브랜치마다 적용하면 병합 결과에 N번 더해져 틀리므로,
               // 브랜치 0만 원본을 받고 나머지 브랜치는 NoValue를 받는다.
  } kind = Kind::Partition;
};

/// propagate/planWeightSplit이 반환하는 답: "분할된 값이 이 op를 통과하면
/// 무슨 일이 일어나는가"를 두 부분으로 서술한다.
struct OpSplitStep {
  /// op의 result 하나당 분할 상태 하나(result 순서와 동일). 대부분의 ONNX
  /// op는 result가 1개이므로 보통 원소도 1개다.
  llvm::SmallVector<SplitState, 1> results;
  /// 위 상태 전이가 성립하려면 엔진이 추가로 잘라 줘야 하는 operand 목록.
  /// 필요 없으면 빈 벡터(예: 단항 elementwise는 아무것도 요구하지 않음).
  llvm::SmallVector<SliceReq, 2> slices;
};

/// op 하나의 분할 특성. 모든 메서드는 순수(IR 변경 없음) — 유일한 예외인
/// patchClone도 엔진이 만들어 준 클론의 attribute/보조 상수만 보정한다.
class SplitRule {
public:
  virtual ~SplitRule() = default;

  /// PLAN(순방향 전파): operand들의 분할 상태가 주어졌을 때 result의 분할
  /// 상태와 추가 슬라이스 요구를 답한다. failure = 이 op는 이 분할을 소화할
  /// 수 없음 → 후보 기각.
  virtual mlir::FailureOr<OpSplitStep> propagate(mlir::Operation *op,
      llvm::ArrayRef<SplitState> operands, int nBranches) const = 0;

  /// PLAN(S 전용): 자신의 "상수" operand(weight/bias)를 잘라 result가
  /// 파티션되게 만드는 계획. 유도되는 축은 op가 결정한다(Conv → out-channel,
  /// MatMul → weight의 M/N). 상수의 Split은 상수폴딩으로 접혀 런타임 비용이
  /// 없다. 시드는 이 방법뿐이다(S는 weight-split만 허용).
  /// failure = weight-split 불가 → 후보 기각.
  virtual mlir::FailureOr<OpSplitStep> planWeightSplit(
      mlir::Operation *op, int nBranches) const {
    return mlir::failure();
  }

  /// REWRITE: 엔진이 클론과 operand 연결(mapping + 슬라이스 배선)을 끝낸
  /// 뒤, SliceReq로 표현할 수 없는 마무리를 한다.
  /// 예: depthwise Conv의 group 재계산, Reshape shape 상수의 dim값/N.
  virtual void patchClone(mlir::OpBuilder &builder, mlir::Operation *cloned,
      int branchIdx, int nBranches, const OpSplitStep &step) const {}
};

/// "op 종류 → 그 op의 규칙" 대응표. 3단 폴백으로 op-agnostic함을 완성한다:
///   ① op 전용 규칙 → ② eltwise 카테고리 규칙(unary/binary-broadcast)
///   → ③ nullptr = 모르는 op → 후보 기각(조용히 틀리는 대신 안전하게 스킵).
/// 규칙 객체들은 상태(필드)가 없으므로 종류당 하나만 만들어 전부가 공유한다.
class SplitRuleRegistry {
public:
  /// 프로그램 전체에 하나뿐인 대응표를 반환한다(최초 호출 시 1회 생성).
  static const SplitRuleRegistry &instance();

  /// op에 적용할 규칙을 ①→②→③ 순서로 찾는다. 모르는 op면 nullptr를
  /// 반환하며, 호출자는 그 후보를 기각해야 한다.
  const SplitRule *lookup(mlir::Operation *op) const;

private:
  /// 대응표는 instance()를 통해서만 만들어진다. 생성자가 모든 규칙을 등록.
  SplitRuleRegistry();

  /// op 이름(예: "onnx.Conv") → 전용 규칙.
  llvm::StringMap<const SplitRule *> exactRules;
  /// 전용 규칙이 없는 op를 위한 카테고리 규칙: 분할 상태를 그대로
  /// 통과시키는 elementwise 부류 (unary: Relu/Sigmoid/Gelu…,
  /// binary: Add/Mul… — broadcast 상수 슬라이스 처리 포함).
  const SplitRule *eltwiseUnaryRule = nullptr;
  const SplitRule *eltwiseBinaryRule = nullptr;
};

} // namespace peakmem
} // namespace onnx_mlir
