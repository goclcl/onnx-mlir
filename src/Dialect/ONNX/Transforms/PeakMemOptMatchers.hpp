//===------ PeakMemOptMatchers.hpp - 패턴 matcher ------===//
//
// 피크 op 주변에서 분할 후보 서브그래프를 찾는 세 matcher.
// 전부 read-only(IR 변경 없음)이며, 크기/토폴로지만 보므로 op-agnostic이다.
//  - expand/shrink: 단일 체인 — 텐서가 커졌다가(EXPAND) 다시 작아지는(SHRINK)
//    구간. 영역 내부에 fork가 있으면 매칭하지 않는다(fork/join의 소관).
//  - fork/join: 한 op의 출력이 여러 op에서 쓰이고 브랜치가 다시 합류하는 DAG.
//  - merge-shrink: 기존 Concat의 결과가 shrink되는 구간(S = Concat).
//
//===----------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/Transforms/PeakMemOptUtils.hpp"

namespace onnx_mlir {
namespace peakmem {

Subgraph matchExpandShrinkPattern(
    mlir::Operation *peakOp, uint32_t detectionScope);

Subgraph matchForkJoinPattern(mlir::Operation *peakOp);

Subgraph matchMergeShrinkPattern(mlir::Operation *peakOp);

} // namespace peakmem
} // namespace onnx_mlir
