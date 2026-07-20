//===------ PeakMemOptUtils.hpp - 공용 타입/헬퍼 ------===//
//
// PeakMemOpt 파일들(Matchers/Planner/Engine/Pass)이 공유하는 타입과 헬퍼.
//  - Subgraph: S(시작 op)부터 E(끝 op)까지로 닫히는 분할 대상 영역
//  - 텐서 크기/liveness 계산, 도달성 탐색, 고립성 검사
//  - pmoDbg(): 디버그 스트림. 기본은 침묵이며, 디버그 빌드에서
//    --debug-only=peak-mem-opt 를 줄 때만 stderr로 출력된다.
//
//===----------------------------------------------------------------===//

#pragma once

#include <deque>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace onnx_mlir {
namespace peakmem {

/// 디버그 스트림. 기본은 아무것도 출력하지 않는다.
llvm::raw_ostream &pmoDbg();

/// Operation의 입력 대비 출력 텐서 크기 변화
enum class TensorSizeChange {
  EXPAND, // 크기가 커짐
  SHRINK, // 크기가 작아짐
  SAME    // 크기가 동일함
};

enum class IsolateStatus {
  None,
  Isolated,
  ExternalInput,
  ExternalOutput,
  NotSplittable
};

const char *toString(TensorSizeChange v);
const char *toString(IsolateStatus v);

/// s에서 시작해 e로 끝나는 서브 그래프 노드 집합(토폴로지 정렬).
llvm::SetVector<mlir::Operation *> getSubgraphNodes(
    mlir::Operation *s, mlir::Operation *e);

/// 분할 대상 영역. subgraphNodes는 S..E의 모든 op를 토폴로지 순서로 담는다
/// (front == S, back == E).
struct Subgraph {
  llvm::SetVector<mlir::Operation *> subgraphNodes;

  Subgraph() = default;
  Subgraph(mlir::Operation *s, mlir::Operation *e) {
    subgraphNodes = getSubgraphNodes(s, e);
  }

  mlir::Operation *getS() {
    return subgraphNodes.empty() ? nullptr : subgraphNodes.front();
  }
  const mlir::Operation *getS() const {
    return subgraphNodes.empty() ? nullptr : subgraphNodes.front();
  }
  mlir::Operation *getE() {
    return subgraphNodes.empty() ? nullptr : subgraphNodes.back();
  }
  const mlir::Operation *getE() const {
    return subgraphNodes.empty() ? nullptr : subgraphNodes.back();
  }
};

/// 상수에서만 유도된 값(상수 자체, 상수의 Split 조각 등)인가. 이런 값은
/// 상수폴딩으로 컴파일 타임에 접혀 바이너리에 들어가므로 런타임 텐서
/// 메모리로 세지 않는다.
bool isFoldedOffline(mlir::Value v);

/// 텐서의 바이트 크기. none/오프라인 값은 0, unranked/동적 차원은 -1.
int64_t getTensorSize(mlir::Value tensor);

/// 지정 op 시점에 라이브인 intermediate 텐서들의 바이트 합(상수류 제외).
int64_t getLiveTensorsSize(mlir::Operation *op, mlir::Liveness &liveness);

/// op의 입력 대비 출력 텐서 크기 변화를 분류.
TensorSizeChange checkTensorSizeChange(mlir::Operation *op);

/// S에서 E까지의 순방향 도달 노드 집합 (E의 사용자는 탐색하지 않음).
llvm::DenseSet<mlir::Operation *> getForwardReachableNodes(
    mlir::Operation *s, mlir::Operation *e);

/// E에서 S까지의 역방향 도달 노드 집합 (S의 입력은 탐색하지 않음).
llvm::DenseSet<mlir::Operation *> getBackwardReachableNodes(
    mlir::Operation *s, mlir::Operation *e);

/// 서브그래프 고립성 검사: S 외 노드의 외부 입력 / E 외 노드의 외부 출력이
/// 없으면 Isolated.
IsolateStatus checkIsolation(const Subgraph &subgraph);

/// 디버그용 한 줄 op 출력 (pmoDbg 스트림으로).
void printOpOneLine(mlir::Operation *op);

} // namespace peakmem
} // namespace onnx_mlir
