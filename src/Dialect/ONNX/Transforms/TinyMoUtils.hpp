//===------ TinyMoUtils.hpp - TinyMo 공용 유틸/엔진 ------===//
//
// TinyMo(baseline) 재현 구현이 공유하는 유틸과 두 최적화 엔진.
//  - tensor splitting: separable conv(pw→dw)의 중간 텐서를 채널로 2분할
//    (TensorSplittingPass의 로직을 함수로 추출한 것)
//  - tensor spilling: 피크 시점에 잠자는(long-living) 텐서를 cold range
//    동안 스토리지로 내렸다가(fetch) 되가져옴. spill/fetch는 ONNXCustomOp
//    (function_name=om_spill/om_fetch/om_fetch_concat2)로 표현하고, 런타임
//    함수는 onnxmlirmodels/tinymo/spill_rt.c 가 제공한다.
//
//===----------------------------------------------------------------===//

#pragma once

#include <optional>

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {
namespace tinymo {

/// 정적 TensorType의 바이트 크기. 미지원/동적이면 0 (TensorSplittingPass의
/// 관례를 따름 — 동적 dim은 1로 세지 않고 전체를 0 처리하지 않도록 기존
/// 구현과 동일하게 동적이면 1을 반환한다).
int64_t getTensorSize(mlir::TensorType tensorType);

/// op 시점 라이브 텐서 바이트 합. includeConstants=false면 onnx.Constant가
/// 정의한 값(파라미터)을 제외한다 — TinyMo의 계정(파라미터는 플래시 상주,
/// activation만 RAM).
int64_t memoryUsageAtOp(mlir::Operation *op, mlir::Liveness &liveness,
    bool includeConstants);

//===--------------------- tensor splitting ---------------------===//

/// 분할 가능한 separable conv 패턴: pw Conv → (Clip) → dw Conv,
/// 중간 텐서가 dw 출력보다 클 때.
struct SplitCandidate {
  mlir::ONNXConvOp pwConv;
  mlir::ONNXClipOp clip; // 없으면 null
  mlir::ONNXConvOp dwConv;
  int64_t interBytes = 0; // pw 중간 텐서 크기
};

/// peakOp의 첫 operand 생산자에서 분할 패턴을 찾는다 (TensorSplittingPass의
/// isOptimizable과 동일한 판별).
std::optional<SplitCandidate> findSplitCandidate(mlir::Operation *peakOp);

/// 기대 피크 감소량: 중간 텐서의 절반 (채널 2분할).
int64_t splitExpectedReduction(const SplitCandidate &cand);

/// 분할 적용 (TensorSplittingPass::performTensorSplitting과 동일 변환).
bool applyTensorSplitting(SplitCandidate &cand, mlir::OpBuilder &builder);

//===--------------------- tensor spilling ---------------------===//

/// spill 대상: 피크 시점에 라이브지만 피크 op가 접근하지 않는 텐서 중
/// cold range(접근 없는 구간)가 가장 긴 것.
struct SpillCandidate {
  mlir::Value victim;
  mlir::Operation *spillAfter = nullptr;  // cold range 시작(마지막 접근 op)
  mlir::Operation *fetchBefore = nullptr; // cold range 끝(다음 접근 op)
  int64_t coldLen = 0;                    // op 개수 기준 구간 길이
  int64_t bytes = 0;                      // victim 크기 = 기대 피크 감소량
  // fetch를 소비자 Concat(2~4입력)과 융합 가능한가. victim이 피크 op에서
  // 소비되는 경우(skip을 모으는 Concat이 피크인 U-Net류)는 융합일 때만
  // 피크 감소 효과가 있다 — fetched 텐서가 실체화되지 않기 때문.
  bool fuse = false;
};

/// 피크 시점 live 텐서들의 cold range를 계산해 최장 후보를 찾는다.
/// 상수/블록 인자/None/피크에서 접근되는 값은 제외.
std::optional<SpillCandidate> findSpillCandidate(
    mlir::func::FuncOp funcOp, mlir::Liveness &liveness, mlir::Operation *peakOp);

/// 기대 피크 감소량: victim 크기 (cold range 동안 라이브 집합에서 빠짐).
inline int64_t spillExpectedReduction(const SpillCandidate &cand) {
  return cand.bytes;
}

/// spill 적용 시의 예상 전역 피크: cold range 내부의 op는 usage-bytes,
/// 바깥은 usage 그대로의 최댓값. 같은 피크값 지점이 여럿이면 spill 하나로
/// 전역 피크가 안 내려갈 수 있으므로, 적용 전에 이걸로 게이트한다.
int64_t expectedPeakAfterSpill(mlir::func::FuncOp funcOp,
    mlir::Liveness &liveness, const SpillCandidate &cand);

/// split 적용 시의 예상 전역 피크: pw 중간 텐서가 라이브인 구간(pw~dw)의
/// op는 usage - interBytes/2, 바깥은 그대로의 최댓값 (코스 추정).
int64_t expectedPeakAfterSplit(mlir::func::FuncOp funcOp,
    mlir::Liveness &liveness, const SplitCandidate &cand);

/// spill/fetch 삽입. victim의 다음 접근이 2-입력 Concat이고 그 뒤 재사용이
/// 없으면 fetch를 concat과 융합(om_fetch_concat2)해 fetched 중간 텐서를
/// 실체화하지 않는다 (논문 Fig. 4(c)). spillId는 호출자가 증가시킨다.
bool applyTensorSpilling(
    SpillCandidate &cand, mlir::OpBuilder &builder, int64_t spillId);

} // namespace tinymo
} // namespace onnx_mlir
