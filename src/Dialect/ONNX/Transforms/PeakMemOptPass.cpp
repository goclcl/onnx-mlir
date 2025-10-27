#include <algorithm>
#include <deque>
#include <queue>
#include <system_error>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {
struct PeakMemOptPass
    : public PassWrapper<PeakMemOptPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PeakMemOptPass)

  StringRef getArgument() const override { return "onnx-peak-mem-opt"; }
  StringRef getDescription() const override {
    return "Performs graph transformations to lower the peak memory usage.";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    Liveness &liveness = getAnalysis<Liveness>();

    /* ===============Peak 탐색=============== */
    Operation *peakOp = nullptr;
    int64_t peakBytes = -1;

    funcOp.walk([&](Operation *op) {
      // onnx.Constant와 func.func op는 스킵
      if (isa<ONNXConstantOp>(op) || isa<func::FuncOp>(op))
        return;

      int64_t liveTensorsSize = getLiveTensorsSize(op, liveness);

      if (liveTensorsSize > peakBytes) {
        peakBytes = liveTensorsSize;
        peakOp = op;
      }
    });

    if (!peakOp) {
      llvm::outs() << "[PeakMemOptPass] No operation found.\n";
      return;
    }

    llvm::outs() << "[peak]" << peakBytes << "B at ";
    printOpOneLine(peakOp);
    llvm::outs() << "\n";

    /* ===============Optimizable Patterns 탐색=============== */
    Operation *s = nullptr;
    Operation *e = nullptr;
    const uint32_t DETECTION_SCOPE = 64;

    llvm::outs() << "[match] try EXPAND~SHRINK from peak, scope="
                 << DETECTION_SCOPE << "\n";

    bool matched = matchExpandShrinkPattern(peakOp, s, e, DETECTION_SCOPE);

    if (!matched) {
      llvm::outs() << "[match] no window\n";
    } else {
      llvm::outs() << "[match] SPLITTABLE window: S=";
      printOpOneLine(s);
      llvm::outs() << "  E=";
      printOpOneLine(e);
      llvm::outs() << "\n";
    }
  }

private:
  // 텐서 크기 변화 상태를 나타내는 열거형
  enum class TensorSizeChange {
    EXPAND, // 크기가 커짐
    SHRINK, // 크기가 작아짐
    SAME    // 크기가 동일함
  };

  enum class IsolateStatus {
    None,
    Splittable,
    ExternalInput,
    ExternalOutput,
    NotSplittable
  };

  // TensorType 크기 계산: byte단위의 사이즈 반환
  static int64_t getTensorSize(Value tensor) {
    Operation *defOp = tensor.getDefiningOp();

    // defining op가 constant거나 noValue면 0 반환
    if (!isa<BlockArgument>(tensor)) { // null casting 방지
      if (isa<ONNXConstantOp>(defOp) || isa<ONNXNoneOp>(defOp))
        return 0;
    }

    // TensorType이 아닐 때
    TensorType tt;
    if (!(tt = dyn_cast<TensorType>(tensor.getType())))
      return 0;

    Type elementType = tt.getElementType();

    unsigned bitWidth = elementType.getIntOrFloatBitWidth();

    int64_t total = 1;
    for (int64_t d : tt.getShape()) {
      if (d == ShapedType::kDynamic) {
        // 동적 차원 발견 시 에러
        llvm::errs()
            << "[PeakMemOptPass] Error: dynamic dimension found in type: " << tt
            << "\n";
        return -1;
      }
      total *= d;
    }
    return (total * bitWidth) / 8;
  }

  // 지정 Op 시점 라이브 intermediate tensor들의 바이트 합 (상수 제외)
  static int64_t getLiveTensorsSize(Operation *op, Liveness &liveness) {
    const LivenessBlockInfo *blk = liveness.getLiveness(op->getBlock());
    if (!blk) {
      llvm::outs() << "[getLiveTensorsSize] !blk" << '\n';
      return 0;
    }

    int64_t bytes = 0;
    SmallPtrSet<Value, 16> live;

    // live values
    for (Value v : blk->currentlyLiveValues(op)) {
      live.insert(v);
    }

    // 현재 op의 결과 values
    for (Value r : op->getResults()) {
      live.insert(r);
    }
    for (Value v : live) {
      if (Operation *def = v.getDefiningOp()) {
        if (isa<ONNXConstantOp>(def))
          continue; // 상수 제외
      }
      bytes += getTensorSize(v);
    }
    return bytes;
  }

  // Operation의 입력 대비 출력 텐서 크기 변화를 확인
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

    // [outs] 간단 요약
    llvm::outs() << "[size] ";
    printOpOneLine(op);
    llvm::outs() << " in=" << totalInputSize << "B"
                 << " out=" << totalOutputSize << "B"
                 << " -> " << toString(res) << "\n";

    return res;
  }

  // --- 헬퍼 함수 1: S에서 E까지만 순방향 탐색 ---
  // S에서 시작하여 도달 가능한 모든 노드 집합을 반환합니다.
  // E 노드를 만나면, E는 집합에 포함하되 E의 사용자는 더 이상 탐색하지
  // 않습니다.
  static llvm::DenseSet<mlir::Operation *> getForwardReachableNodes(
      Operation *S, Operation *E) {
    llvm::DenseSet<mlir::Operation *> visitedSet;
    std::deque<mlir::Operation *> worklist;

    worklist.push_back(S);
    visitedSet.insert(S);

    while (!worklist.empty()) {
      mlir::Operation *currentOp = worklist.front();
      worklist.pop_front();

      // E에 도달하면, E의 자식(user) 노드들은 탐색 큐에 넣지 않습니다.
      if (currentOp == E) {
        continue;
      }

      for (mlir::Value result : currentOp->getResults()) {
        for (mlir::Operation *userOp : result.getUsers()) {
          // 이미 방문한 노드는 스킵
          if (visitedSet.contains(userOp)) {
            continue;
          }
          visitedSet.insert(userOp);
          worklist.push_back(userOp);
        }
      }
    }
    return visitedSet;
  }

  // --- 헬퍼 함수 2: E에서 S까지만 역방향 탐색 ---
  // E에서 시작하여 E에 도달하는 모든 노드 집합을 반환합니다.
  // S 노드를 만나면, S는 집합에 포함하되 S의 입력(operand)은 더 이상 탐색하지
  // 않습니다.
  static llvm::DenseSet<mlir::Operation *> getBackwardReachableNodes(
      mlir::Operation *S, mlir::Operation *E) {
    llvm::DenseSet<mlir::Operation *> visitedSet;
    std::deque<mlir::Operation *> worklist;

    worklist.push_back(E);
    visitedSet.insert(E);

    while (!worklist.empty()) {
      mlir::Operation *currentOp = worklist.front();
      worklist.pop_front();

      // S에 도달하면, S의 부모(operand) 노드들은 탐색 큐에 넣지 않습니다.
      if (currentOp == S) {
        continue;
      }

      for (mlir::Value operand : currentOp->getOperands()) {
        mlir::Operation *defOp = operand.getDefiningOp();

        // Block Argument이거나 이미 방문한 노드는 스킵
        // constant와 noValue op도 스킵
        if (!defOp || visitedSet.contains(defOp) ||
            isa<ONNXConstantOp>(defOp) || isa<ONNXNoneOp>(defOp)) {
          continue;
        }

        visitedSet.insert(defOp);
        worklist.push_back(defOp);
      }
    }
    return visitedSet;
  }

  // --- 메인 함수: 서브그래프 고립성 검사 ---
  static IsolateStatus isIsolatedSubGraph(
      mlir::Operation *S, mlir::Operation *E) {

    // S나 E를 못찾았으면 notSplittable
    if (!S || !E) {
      llvm::outs() << "[isIsolatedSubGraph] NotSplittable: S or E is null\n";
      return IsolateStatus::NotSplittable;
    }

    // 1단계: 서브그래프 노드 집합 정의
    llvm::DenseSet<mlir::Operation *> forwardSet =
        getForwardReachableNodes(S, E);

    llvm::DenseSet<mlir::Operation *> backwardSet =
        getBackwardReachableNodes(S, E);

    llvm::DenseSet<mlir::Operation *> subGraphNodes;
    for (mlir::Operation *op : forwardSet) {
      if (backwardSet.contains(op)) {
        subGraphNodes.insert(op);
      }
    }

    // (방어 코드) S와 E가 교집합에 모두 포함되어 있어야 함
    if (!subGraphNodes.contains(S) || !subGraphNodes.contains(E)) {
      llvm::outs()
          << "[isIsolatedSubGraph] Failed: Path intersection is incomplete. "
          << "S (" << S->getName() << ") or E (" << E->getName()
          << ") is not included in the final subgraph.\n";
      return IsolateStatus::NotSplittable;
    }

    // 2단계: I/O 검사
    for (mlir::Operation *op : subGraphNodes) {

      // --- 2.1 External Input 검사 ---
      if (op != S) {
        for (mlir::Value operand : op->getOperands()) {
          mlir::Operation *defOp = operand.getDefiningOp();

          // constant와 noValue op는 스킵
          if (isa<ONNXConstantOp>(defOp) || isa<ONNXNoneOp>(defOp)) {
            continue;
          }

          if (!defOp) {
            // Case 1: BlockArgument (S가 아닌데 함수 인자를 받음)
            llvm::outs()
                << "[isIsolatedSubGraph] Failed: External input. Node ("
                << op->getName() << ") is not S, but it receives "
                << "a BlockArgument (" << operand << ") as input.\n";
            return IsolateStatus::NotSplittable;
          }

          if (!subGraphNodes.contains(defOp)) {
            // Case 2: Op Result (S 또는 서브그래프 내부 노드가 아닌 곳에서
            // 입력을 받음)
            llvm::outs() << "[isIsolatedSubGraph] ExternalInput at ";
            printOpOneLine(op);
            llvm::outs() << " <- ";
            printOpOneLine(defOp);
            llvm::outs() << "\n";
            return IsolateStatus::ExternalInput;
          }
        }
      }

      // --- 2.2 External Output 검사 ---
      if (op != E) {
        for (mlir::Value result : op->getResults()) {
          for (mlir::Operation *userOp : result.getUsers()) {
            // 서브그래프 외부(E가 아닌) 노드가 이 값을 사용함
            if (!subGraphNodes.contains(userOp)) {
              llvm::outs() << "[isIsolatedSubGraph] ExternalOutput at ";
              printOpOneLine(op);
              llvm::outs() << " -> ";
              printOpOneLine(userOp);
              llvm::outs() << "\n";
              return IsolateStatus::ExternalOutput;
            }
          }
        }
      }
    }

    // 모든 검사를 통과
    return IsolateStatus::Splittable;
  }

  bool matchExpandShrinkPattern(Operation *peakOp, Operation *&s, Operation *&e,
      uint32_t detectionScope) {
    Operation *expander = nullptr;
    Operation *shrinker = nullptr;
    SmallPtrSet<Operation *, 32> visited;
    std::queue<Operation *> backwardWorklist;
    std::queue<Operation *> forwardWorklist;
    bool foundExpander = false;
    bool foundShrinker = false;
    IsolateStatus status = IsolateStatus::None;

    // [outs] 시작
    llvm::outs() << "[match] start: peak=";
    printOpOneLine(peakOp);
    llvm::outs() << " scope=" << detectionScope << "\n";

    // peakOp를 wokrlist에 추가
    if (visited.insert(peakOp).second) {
      backwardWorklist.push(peakOp);
      forwardWorklist.push(peakOp);
    }

    auto backwardSearch = [&]() {
      // 기존 후보 S가 존재했다면 초기화
      if (foundExpander) {
        expander = nullptr;
        foundExpander = false;
      }

      while (detectionScope > 0 && !backwardWorklist.empty()) {
        Operation *currentOp = backwardWorklist.front(); // 맨 앞 원소 확인
        if (!currentOp)
          continue;
        backwardWorklist.pop(); // 맨 앞 원소 제거

        // constant 또는 noValue라면 skip
        if (isa<ONNXConstantOp>(currentOp) || isa<ONNXNoneOp>(currentOp)) {
          continue;
        } else
          detectionScope--;

        llvm::outs() << "[bwd] pop ";
        printOpOneLine(currentOp);
        llvm::outs() << " scopeLeft=" << detectionScope << "\n";

        for (Value operand : currentOp->getOperands()) {
          if (Operation *defOp = operand.getDefiningOp()) {
            if (visited.insert(defOp).second)
              backwardWorklist.push(defOp); // 상위 op를 worklist에 추가
          }
        }

        if (checkTensorSizeChange(currentOp) == TensorSizeChange::EXPAND) {
          expander = currentOp;
          foundExpander = true;
          llvm::outs() << "[bwd] EXPAND candidate: ";
          printOpOneLine(expander);
          llvm::outs() << "\n";
          break;
        }
      }
    };

    auto forwardSearch = [&]() {
      // 기존 후보 E가 존재했다면 초기화
      if (foundShrinker) {
        shrinker = nullptr;
        foundShrinker = false;
      }

      while (detectionScope > 0 && !forwardWorklist.empty()) {
        Operation *currentOp = forwardWorklist.front();
        forwardWorklist.pop();

        if (isa<ONNXConstantOp>(currentOp))
          continue;
        else
          detectionScope--;

        llvm::outs() << "[fwd] pop ";
        printOpOneLine(currentOp);
        llvm::outs() << " scopeLeft=" << detectionScope << "\n";

        for (Value result : currentOp->getResults()) {
          for (Operation *user : result.getUsers()) {
            if (visited.insert(user).second)
              forwardWorklist.push(user);
          }
        }

        if (checkTensorSizeChange(currentOp) == TensorSizeChange::SHRINK) {
          shrinker = currentOp;
          foundShrinker = true;
          llvm::outs() << "[fwd] SHRINK candidate: ";
          printOpOneLine(shrinker);
          llvm::outs() << "\n";
          break;
        }
      }
    };

    while (status != IsolateStatus::Splittable) {
      if (detectionScope == 0) {
        llvm::outs() << "[match] scope exhausted\n";
        return false;
      }

      if (status == IsolateStatus::None) {
        backwardSearch();
        forwardSearch();
        status = isIsolatedSubGraph(expander, shrinker);
      } else if (status == IsolateStatus::ExternalInput) {
        llvm::outs() << "[match] status=" << toString(status)
                     << " -> expand backward\n";
        backwardSearch();
        status = isIsolatedSubGraph(expander, shrinker);
      } else if (status == IsolateStatus::ExternalOutput) {
        llvm::outs() << "[match] status=" << toString(status)
                     << " -> expand forward\n";
        forwardSearch();
        status = isIsolatedSubGraph(expander, shrinker);
      } else if (status == IsolateStatus::NotSplittable) {
        llvm::outs() << "[match] NotSplittable\n";
        return false;
      }
      llvm::outs() << "[match] status now=" << toString(status) << "\n";
    }

    // 최종 성공: s/e를 실제로 바깥에 반영
    s = expander;
    e = shrinker;

    llvm::outs() << "[match] SPLITTABLE: S=";
    printOpOneLine(s);
    llvm::outs() << "  E=";
    printOpOneLine(e);
    llvm::outs() << "\n";

    return true;

    // S~E까지 몇 개의 op를 split해야하며
    // 최적화로 얻는 이득은 얼마인지?(peak memory reduction)
  }

  bool matchForkJoinPattern(Operation *op, uint32_t detectionScope) {
    return false;
  }

  bool matchMergeShrinkPattern(Operation *op, uint32_t detectionScope) {
    return false;
  }

  // ----[ DEBUG (outs) helpers ]----
  static void printOpOneLine(Operation *op) {
    if (!op) {
      llvm::outs() << "<null-op>";
      return;
    }
    llvm::outs() << op->getName() << " @" << op->getLoc()
                 << " (in=" << op->getNumOperands()
                 << ", out=" << op->getNumResults() << ")";
  }

  static const char *toString(TensorSizeChange v) {
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

  static const char *toString(IsolateStatus v) {
    switch (v) {
    case IsolateStatus::None:
      return "None";
    case IsolateStatus::Splittable:
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
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createPeakMemOptPass() {
  return std::make_unique<PeakMemOptPass>();
}
} // namespace onnx_mlir
