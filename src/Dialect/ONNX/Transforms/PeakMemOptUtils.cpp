//===------ PeakMemOptUtils.cpp - 공용 타입/헬퍼 구현 ------===//

#include "src/Dialect/ONNX/Transforms/PeakMemOptUtils.hpp"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#define DEBUG_TYPE "peak-mem-opt"

using namespace mlir;

namespace onnx_mlir {
namespace peakmem {

llvm::raw_ostream &pmoDbg() {
#ifndef NDEBUG
  if (llvm::DebugFlag && llvm::isCurrentDebugType(DEBUG_TYPE))
    return llvm::dbgs();
#endif
  static llvm::raw_null_ostream nullOs;
  return nullOs;
}

const char *toString(TensorSizeChange v) {
  switch (v) {
  case TensorSizeChange::EXPAND:
    return "EXPAND";
  case TensorSizeChange::SHRINK:
    return "SHRINK";
  case TensorSizeChange::SAME:
    return "SAME";
  }
  return "?";
}

const char *toString(IsolateStatus v) {
  switch (v) {
  case IsolateStatus::None:
    return "None";
  case IsolateStatus::Isolated:
    return "Splittable";
  case IsolateStatus::ExternalInput:
    return "ExternalInput";
  case IsolateStatus::ExternalOutput:
    return "ExternalOutput";
  case IsolateStatus::NotSplittable:
    return "NotSplittable";
  }
  return "?";
}

void printOpOneLine(Operation *op) {
  if (!op) {
    pmoDbg() << "<null-op>";
    return;
  }
  pmoDbg() << op->getName() << " @" << op->getLoc()
           << " (in=" << op->getNumOperands()
           << ", out=" << op->getNumResults() << ")";
}

bool isFoldedOffline(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<ONNXConstantOp>(def) || isa<ONNXNoneOp>(def))
    return true;
  if (isa<ONNXSplitOp>(def))
    return isFoldedOffline(def->getOperand(0));
  return false;
}

bool isOfflineProducer(Operation *op) {
  return op && op->getNumResults() >= 1 && isFoldedOffline(op->getResult(0));
}

int64_t getTensorSize(Value tensor) {
  Operation *defOp = tensor.getDefiningOp();

  Type type = tensor.getType();

  // value가 none이면 0반환
  if (isa<NoneType>(type))
    return 0;

  // 오프라인으로 접히는 값은 0 반환
  if (isFoldedOffline(tensor))
    return 0;

  RankedTensorType rtt = dyn_cast<RankedTensorType>(type);
  if (!rtt) {
    pmoDbg() << "[getTensorSize] fail: Unranked tensor at ";
    printOpOneLine(defOp);
    pmoDbg() << "\n";
    return -1;
  }

  Type elementType = rtt.getElementType();

  unsigned bitWidth = elementType.getIntOrFloatBitWidth();

  int64_t total = 1;
  for (int64_t d : rtt.getShape()) {
    if (d == ShapedType::kDynamic) {
      // 동적 차원은 크기를 셀 수 없다 (이 패스는 정적 모양 전제)
      llvm::errs()
          << "[getTensorSize] fail: dynamic dimension found in type: " << rtt
          << "\n";
      return -1;
    }
    total *= d;
  }
  return (total * bitWidth) / 8;
}

int64_t getLiveTensorsSize(Operation *op, Liveness &liveness) {
  const LivenessBlockInfo *blk = liveness.getLiveness(op->getBlock());
  if (!blk) {
    pmoDbg() << "[getLiveTensorsSize] fail: !blk" << '\n';
    return 0;
  }

  int64_t bytes = 0;
  llvm::SmallPtrSet<Value, 16> live;

  // live values
  for (Value v : blk->currentlyLiveValues(op)) {
    live.insert(v);
  }

  // 현재 op의 결과 values
  for (Value r : op->getResults()) {
    live.insert(r);
  }
  for (Value v : live) {
    if (isFoldedOffline(v))
      continue; // 상수(및 상수의 Split 조각) 제외
    bytes += getTensorSize(v);
  }
  return bytes;
}

TensorSizeChange checkTensorSizeChange(Operation *op) {
  int64_t totalInputSize = 0;
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();

    if (defOp) {
      if (isa<ONNXConstantOp>(defOp))
        continue;
    }

    int64_t size = getTensorSize(operand);
    totalInputSize += size;
  }

  int64_t totalOutputSize = 0;
  for (Value result : op->getResults()) {
    int64_t size = getTensorSize(result);
    totalOutputSize += size;
  }

  TensorSizeChange res;
  if (totalOutputSize > totalInputSize) {
    res = TensorSizeChange::EXPAND;
  } else if (totalOutputSize < totalInputSize) {
    res = TensorSizeChange::SHRINK;
  } else {
    res = TensorSizeChange::SAME;
  }

  return res;
}

llvm::DenseSet<Operation *> getForwardReachableNodes(
    Operation *s, Operation *e) {
  llvm::DenseSet<Operation *> visitedSet;
  std::deque<Operation *> worklist;

  worklist.push_back(s);
  visitedSet.insert(s);

  while (!worklist.empty()) {
    Operation *currentOp = worklist.front();
    worklist.pop_front();

    // E에 도달하면, E의 자식(user) 노드들은 탐색 큐에 넣지 않음
    if (currentOp == e) {
      continue;
    }

    for (Operation *userOp : currentOp->getUsers()) {
      // 이미 방문한 노드는 스킵
      if (visitedSet.contains(userOp)) {
        continue;
      }
      visitedSet.insert(userOp);
      worklist.push_back(userOp);
    }
  }
  return visitedSet;
}

llvm::DenseSet<Operation *> getBackwardReachableNodes(
    Operation *s, Operation *e) {
  llvm::DenseSet<Operation *> visitedSet;
  std::deque<Operation *> worklist;

  worklist.push_back(e);
  visitedSet.insert(e);

  while (!worklist.empty()) {
    mlir::Operation *currentOp = worklist.front();
    worklist.pop_front();

    // S에 도달하면, S의 부모(operand) 노드들은 탐색 큐에 넣지 않음
    if (currentOp == s) {
      continue;
    }

    for (Value operand : currentOp->getOperands()) {
      Operation *defOp = operand.getDefiningOp();

      // Block Argument(그래프 전체 입력 값)이거나 이미 방문한 노드는 스킵
      // 오프라인으로 접히는 생산자(상수/NoValue/상수의 Split)도 스킵
      if (!defOp || visitedSet.contains(defOp) || isOfflineProducer(defOp)) {
        continue;
      }

      visitedSet.insert(defOp);
      worklist.push_back(defOp);
    }
  }
  return visitedSet;
}

llvm::SetVector<Operation *> getSubgraphNodes(Operation *s, Operation *e) {
  if (!s || !e)
    return {};

  llvm::DenseSet<Operation *> forwardSet = getForwardReachableNodes(s, e);

  llvm::DenseSet<Operation *> backwardSet = getBackwardReachableNodes(s, e);

  llvm::SetVector<Operation *> subgraphNodes;
  for (Operation *op : forwardSet) {
    if (backwardSet.contains(op)) {
      subgraphNodes.insert(op);
    }
  }

  return topologicalSort(subgraphNodes);
}

IsolateStatus checkIsolation(const Subgraph &subgraph) {
  llvm::SetVector<Operation *> subgraphNodes = subgraph.subgraphNodes;
  const Operation *s = subgraph.getS();
  const Operation *e = subgraph.getE();

  // I/O 검사
  for (Operation *op : subgraphNodes) {

    // External Input 검사
    if (op != s) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();

        // 오프라인으로 접히는 값(상수/NoValue/상수의 Split)은 스킵
        if (isFoldedOffline(operand)) {
          continue;
        }

        if (!defOp) {
          // Case 1: BlockArgument (S가 아닌데 함수 인자를 받음)
          pmoDbg() << "[checkIsolation] Failed: External input. Node ("
                   << op->getName() << ") is not S, but it receives "
                   << "a BlockArgument (" << operand << ") as input.\n";
          return IsolateStatus::NotSplittable;
        }

        if (!subgraphNodes.contains(defOp)) {
          // Case 2: Op Result (S 또는 서브그래프 내부 노드가 아닌 곳에서
          // 입력을 받음)
          pmoDbg() << "[checkIsolation] ExternalInput at ";
          printOpOneLine(op);
          pmoDbg() << " <- ";
          printOpOneLine(defOp);
          pmoDbg() << "\n";
          return IsolateStatus::ExternalInput;
        }
      }
    }

    // --- External Output 검사 ---
    if (op != e) {
      for (mlir::Value result : op->getResults()) {
        for (mlir::Operation *userOp : result.getUsers()) {
          // 서브그래프 외부(E가 아닌) 노드가 이 값을 사용함
          if (!subgraphNodes.contains(userOp)) {
            pmoDbg() << "[checkIsolation] ExternalOutput at ";
            printOpOneLine(op);
            pmoDbg() << " -> ";
            printOpOneLine(userOp);
            pmoDbg() << "\n";
            return IsolateStatus::ExternalOutput;
          }
        }
      }
    }
  }

  // 모든 검사를 통과
  return IsolateStatus::Isolated;
}

} // namespace peakmem
} // namespace onnx_mlir
