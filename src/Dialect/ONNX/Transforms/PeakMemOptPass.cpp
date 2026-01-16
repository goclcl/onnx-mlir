#include <cmath>
#include <cstdint>
#include <deque>
#include <queue>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

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

    // auto &domInfo = getAnalysis<DominanceInfo>();
    // auto &postDomInfo = getAnalysis<PostDominanceInfo>();

    /* ===============Peak 탐색=============== */
    llvm::outs()
        << "\n================ 1. Search for Peak Operation ================\n";
    llvm::SmallVector<Operation *, 8> peakOps;
    // 최적화 대상에서 제외할 ops
    DenseSet<Operation *> coveredOps;
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

    llvm::outs() << "[peak]" << peakBytes << "B at \n";
    for (Operation *peakOp : peakOps) {
      llvm::outs() << "    ";
      printOpOneLine(peakOp);
      llvm::outs() << "\n";
    }

    for (Operation *peakOp : peakOps) {
      // 중복 처리 방지
      if (coveredOps.contains(peakOp)) {
        continue;
      }
      /* ===============Optimizable Patterns 탐색=============== */
      llvm::outs() << "\n================ 2. Detect Optimizable Patterns "
                      "================\n";

      /* ---------------Expand/Shrink--------------- */
      const uint32_t DETECTION_SCOPE = 64;

      subgraph expandShrinkSubgraph =
          matchExpandShrinkPattern(peakOp, DETECTION_SCOPE);

      if (!expandShrinkSubgraph.getS()) {
        llvm::outs() << "[detect Expand/Shrink] no matched pattern\n";
      } else {
        llvm::outs() << "[detect Expand/Shrink] splittable subgraph: S=";
        printOpOneLine(expandShrinkSubgraph.getS());
        llvm::outs() << "  E=";
        printOpOneLine(expandShrinkSubgraph.getE());
        llvm::outs() << "\n";
      }
      /* ---------------Expand/Shrink--------------- */

      /* ---------------Fork/Join--------------- */
      IsolatedSubgraph isolatedSubgraph;

      subgraph forkJoinSubgraph = matchForkJoinPattern(peakOp);

      if (!forkJoinSubgraph.getS()) {
        llvm::outs() << "[detect Fork-Join] no matched pattern\n";
      } else {
        llvm::outs() << "[detect Fork-Join] splittable subgraph: S=";
        printOpOneLine(forkJoinSubgraph.getS());
        llvm::outs() << "  E=";
        printOpOneLine(forkJoinSubgraph.getE());
        llvm::outs() << "\n";
      }
      /* ---------------Fork/Join--------------- */

      /* ---------------Merge/Shrink--------------- */
      subgraph mergeShrinkSubgraph = matchMergeShrinkPattern(peakOp);

      if (mergeShrinkSubgraph.getS()) {
        llvm::outs() << "[detect Merge/Shrink] splittable subgraph: S=";
        printOpOneLine(mergeShrinkSubgraph.getS());
        llvm::outs() << "  E=";
        printOpOneLine(mergeShrinkSubgraph.getE());
        llvm::outs() << "\n";
      } else {
        llvm::outs() << "[detect Merge/Shrink] no matched pattern\n";
      }
      /* ---------------Merge/Shrink--------------- */

      /* ===============Check Splittability=============== */
      llvm::DenseSet<int64_t> exShSplittableDims;
      /* expand/shrink pattern */
      if (expandShrinkSubgraph.getS()) {
        exShSplittableDims = getSplittableDims(expandShrinkSubgraph);
      }

      /* fork/join pattern */
      // if (forkJoinSubgraph.getS()) {
      //   // TODO:
      // }

      /* merge/shrink pattern */
      // if (auto s = mergeShrinkSubgraph.getS()) {
      //   // TODO:
      //   ONNXConcatOp concatOp = dyn_cast<ONNXConcatOp>(s);
      //   int64_t splitDim = concatOp.getAxis();
      //   if (!isDimPreserved(s, splitDim, mergeShrinkSubgraph)) {
      //     llvm::outs()
      //         << "[split Merge/Shrink] Concat dimension is not
      //         splittable.\n";
      //   } else {
      //     optConcat(mergeShrinkSubgraph);
      //   }
      // }

      /* ===============Check Splittability=============== */

      /* ===============Split=============== */
      // splitSubgraph(expandShrinkSubgraph, *exShSplittableDims.begin());
      splitSubgraph(expandShrinkSubgraph, 1);

      // 중복 처리를 피하기 위해 최적화된 ops는 coverdOps에 추가
      auto nodes = expandShrinkSubgraph.subgraphNodes;
      coveredOps.reserve(coveredOps.size() + nodes.size());
      coveredOps.insert(nodes.begin(), nodes.end());
      /* ===============Split=============== */
    }
  }

private:
  // 텐서 크기 변화 상태
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

  struct subgraph {
    llvm::SetVector<Operation *>
        subgraphNodes;           // All ops in s→e path including s and e
    int64_t splitDimension = -1; // Dimension to split along

    // constructors
    subgraph() = default;

    subgraph(Operation *s, Operation *e) {
      subgraphNodes = getSubgraphNodes(s, e);
    }

    Operation *getS() {
      return subgraphNodes.empty() ? nullptr : subgraphNodes.front();
    }
    const Operation *getS() const {
      return subgraphNodes.empty() ? nullptr : subgraphNodes.front();
    }

    Operation *getE() {
      return subgraphNodes.empty() ? nullptr : subgraphNodes.back();
    }
    const Operation *getE() const {
      return subgraphNodes.empty() ? nullptr : subgraphNodes.back();
    }
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
            << "[PeakMemOptPass] fail: dynamic dimension found in type: " << tt
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
      llvm::outs() << "[getLiveTensorsSize] fail: !blk" << '\n';
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

    return res;
  }

  // S에서 E까지만 순방향 탐색
  // S에서 시작하여 도달 가능한 모든 노드 집합을 반환
  // E 노드를 만나면, E는 집합에 포함하되 E의 사용자는 더 이상 탐색하지 않음
  static llvm::DenseSet<Operation *> getForwardReachableNodes(
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

  // E에서 S까지만 역방향 탐색
  // E에서 시작하여 S에 도달하는 모든 노드 집합을 반환
  // S 노드를 만나면, S는 집합에 포함하되 S의 입력(operand)은 더 이상 탐색하지
  // 않음
  static llvm::DenseSet<Operation *> getBackwardReachableNodes(
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

  // s에서 시작해 e로 끝나는 서브 그래프 노드 집합 정렬해서 반환
  static llvm::SetVector<Operation *> getSubgraphNodes(
      Operation *s, Operation *e) {
    if (!s) {
      llvm::outs() << "[getSubgraphNodes] fail: s = nullptr \n";
      return {};
    }
    if (!e) {
      llvm::outs() << "[getSubgraphNodes] fail: e = nullptr \n";
      return {};
    }

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

  // 서브그래프 고립성 검사
  static IsolateStatus checkIsolation(const subgraph &subgraph) {
    llvm::SetVector<Operation *> subgraphNodes = subgraph.subgraphNodes;
    const Operation *s = subgraph.getS();
    const Operation *e = subgraph.getE();

    // I/O 검사
    for (Operation *op : subgraphNodes) {

      // External Input 검사
      if (op != s) {
        for (Value operand : op->getOperands()) {
          Operation *defOp = operand.getDefiningOp();

          // constant와 noValue op는 스킵
          if (isa<ONNXConstantOp>(defOp) || isa<ONNXNoneOp>(defOp)) {
            continue;
          }

          if (!defOp) {
            // Case 1: BlockArgument (S가 아닌데 함수 인자를 받음)
            llvm::outs() << "[checkIsolation] Failed: External input. Node ("
                         << op->getName() << ") is not S, but it receives "
                         << "a BlockArgument (" << operand << ") as input.\n";
            return IsolateStatus::NotSplittable;
          }

          if (!subgraphNodes.contains(defOp)) {
            // Case 2: Op Result (S 또는 서브그래프 내부 노드가 아닌 곳에서
            // 입력을 받음)
            llvm::outs() << "[checkIsolation] ExternalInput at ";
            printOpOneLine(op);
            llvm::outs() << " <- ";
            printOpOneLine(defOp);
            llvm::outs() << "\n";
            return IsolateStatus::ExternalInput;
          }
        }
      }

      // --- 2.2 External Output 검사 ---
      if (op != e) {
        for (mlir::Value result : op->getResults()) {
          for (mlir::Operation *userOp : result.getUsers()) {
            // 서브그래프 외부(E가 아닌) 노드가 이 값을 사용함
            if (!subgraphNodes.contains(userOp)) {
              llvm::outs() << "[checkIsolation] ExternalOutput at ";
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
    return IsolateStatus::Isolated;
  }

  subgraph matchExpandShrinkPattern(
      Operation *peakOp, uint32_t detectionScope) {
    Operation *expander = nullptr;
    Operation *shrinker = nullptr;
    SmallPtrSet<Operation *, 32> visited;
    std::queue<Operation *> backwardWorklist;
    std::queue<Operation *> forwardWorklist;
    bool foundExpander = false;
    bool foundShrinker = false;
    IsolateStatus status = IsolateStatus::None;

    // [outs] 시작
    llvm::outs() << "[matchExpandShrinkPattern] start: peak=";
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

        for (Value operand : currentOp->getOperands()) {
          if (Operation *defOp = operand.getDefiningOp()) {
            if (visited.insert(defOp).second)
              backwardWorklist.push(defOp); // 상위 op를 worklist에 추가
          }
        }

        if (checkTensorSizeChange(currentOp) == TensorSizeChange::EXPAND) {
          expander = currentOp;
          foundExpander = true;
          llvm::outs() << "    [backwardSearch] EXPAND candidate: ";
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

        llvm::outs() << "    [forwardSearch] pop ";
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
          llvm::outs() << "    [forwardSearch] SHRINK candidate: ";
          printOpOneLine(shrinker);
          llvm::outs() << "\n";
          break;
        }
      }
    };

    while (status != IsolateStatus::Isolated) {
      if (detectionScope == 0) {
        llvm::outs() << "[matchExpandShrinkPattern] scope exhausted\n";
        return subgraph{};
      }

      if (status == IsolateStatus::None) {
        backwardSearch();
        forwardSearch();

        subgraph candidateSubgraph = subgraph(expander, shrinker);
        status = checkIsolation(candidateSubgraph);
      } else if (status == IsolateStatus::ExternalInput) {
        llvm::outs() << "[matchExpandShrinkPattern] status=" << toString(status)
                     << " -> expand backward\n";
        backwardSearch();
        status = checkIsolation(subgraph(expander, shrinker));
      } else if (status == IsolateStatus::ExternalOutput) {
        llvm::outs() << "[matchExpandShrinkPattern] status=" << toString(status)
                     << " -> expand forward\n";
        forwardSearch();
        status = checkIsolation(subgraph(expander, shrinker));
      } else if (status == IsolateStatus::NotSplittable) {
        llvm::outs() << "[matchExpandShrinkPattern] NotSplittable\n";
        return subgraph{};
      }
      llvm::outs() << "[matchExpandShrinkPattern] status now="
                   << toString(status) << "\n";
    }

    llvm::outs() << "[matchExpandShrinkPattern] SPLITTABLE: S=";
    printOpOneLine(expander);
    llvm::outs() << "  E=";
    printOpOneLine(shrinker);
    llvm::outs() << "\n";

    return subgraph(expander, shrinker);

    // S~E까지 몇 개의 op를 split해야하며
    // 최적화로 얻는 이득은 얼마인지?(peak memory reduction)
  }

  /*======================================================*/

  struct IsolatedSubgraph {
    Operation *entryGate = nullptr; // S (must be branching; fanout>=2)
    Operation *exitGate = nullptr;  // E
    llvm::SmallVector<Operation *, 256> nodes;
  };

  bool isForkOp(Operation *op) {
    if (op->getNumResults() != 1) {
      llvm::outs() << "[isForkOp] fail: op has multiple results.\n";
    }

    Value result = op->getResult(0);
    return result.hasNUsesOrMore(2);
  }

  bool isJoinOp(Operation *op, llvm::SmallPtrSet<Operation *, 4> branches) {
    for (Operation *branch : branches) {
      llvm::DenseSet<Operation *> nodes = getForwardReachableNodes(branch, op);
      if (!nodes.contains(op))
        return false;
    }

    return true;
  }

  subgraph matchForkJoinPattern(Operation *peakOp) {
    Operation *forkOp = nullptr;
    Operation *joinOp = nullptr;
    bool foundFork = false;
    bool foundJoin = false;
    SmallPtrSet<Operation *, 32> visited;
    std::queue<Operation *> backwardWorklist;
    std::queue<Operation *> forwardWorklist;
    llvm::SmallPtrSet<Operation *, 4> branches;
    IsolateStatus status = IsolateStatus::None;

    auto backwardSearch = [&]() {
      // 기존 후보 S가 존재했다면 초기화
      if (foundFork) {
        forkOp = nullptr;
        foundFork = false;
        branches.clear();
      }

      while (!backwardWorklist.empty()) {
        Operation *currentOp = backwardWorklist.front(); // 맨 앞 원소 확인
        if (!currentOp)
          continue;
        backwardWorklist.pop(); // 맨 앞 원소 제거

        // constant 또는 noValue라면 skip
        if (isa<ONNXConstantOp>(currentOp) || isa<ONNXNoneOp>(currentOp)) {
          continue;
        }

        for (Value operand : currentOp->getOperands()) {
          if (Operation *defOp = operand.getDefiningOp()) {
            if (visited.insert(defOp).second)
              backwardWorklist.push(defOp); // 상위 op를 worklist에 추가
          }
        }

        // 분기점이라면
        if (isForkOp(currentOp)) {
          forkOp = currentOp;
          foundFork = true;
          Value result = forkOp->getResult(0);
          for (Operation *user : result.getUsers()) {
            branches.insert(user);
          }
          llvm::outs() << "    [backwardSearch] forkOp candidate: ";
          printOpOneLine(forkOp);
          llvm::outs() << "\n";
          break;
        }
      }
    };

    auto forwardSearch = [&]() {
      // 기존 후보 E가 존재했다면 초기화
      if (foundJoin) {
        joinOp = nullptr;
        foundJoin = false;
      }

      while (!forwardWorklist.empty()) {
        Operation *currentOp = forwardWorklist.front();
        forwardWorklist.pop();

        if (isa<ONNXConstantOp>(currentOp))
          continue;

        llvm::outs() << "    [forwardSearch] pop ";
        printOpOneLine(currentOp);
        llvm::outs() << "\n";

        for (Value result : currentOp->getResults()) {
          for (Operation *user : result.getUsers()) {
            if (visited.insert(user).second)
              forwardWorklist.push(user);
          }
        }

        // 합류점이라면
        if (isJoinOp(currentOp, branches)) {
          joinOp = currentOp;
          foundJoin = true;
          llvm::outs() << "    [forwardSearch] forkOp candidate: ";
          printOpOneLine(joinOp);
          llvm::outs() << "\n";
          break;
        }
      }
    };

    // [outs] 시작
    llvm::outs() << "[matchForkJoinPattern] start: peak=";
    printOpOneLine(peakOp);
    llvm::outs() << "\n";

    // peakOp를 wokrlist에 추가
    if (visited.insert(peakOp).second) {
      backwardWorklist.push(peakOp);
      forwardWorklist.push(peakOp);
    }

    while (status != IsolateStatus::Isolated) {
      if (status == IsolateStatus::None) {
        backwardSearch();
        forwardSearch();

        subgraph candidateSubgraph = subgraph(forkOp, joinOp);

        status = checkIsolation(candidateSubgraph);
      } else if (status == IsolateStatus::ExternalInput) {
        llvm::outs() << "[matchForkJoinPattern] status=" << toString(status)
                     << " -> expand backward\n";
        backwardSearch();
        status = checkIsolation(subgraph(forkOp, joinOp));
      } else if (status == IsolateStatus::ExternalOutput) {
        llvm::outs() << "[matchForkJoinPattern] status=" << toString(status)
                     << " -> expand forward\n";
        forwardSearch();
        status = checkIsolation(subgraph(forkOp, joinOp));
      } else if (status == IsolateStatus::NotSplittable) {
        llvm::outs() << "[matchForkJoinPattern] NotSplittable\n";
        return subgraph{};
      }
      llvm::outs() << "[matchForkJoinPattern] status now=" << toString(status)
                   << "\n";
    }

    llvm::outs() << "[matchForkJoinPattern] SPLITTABLE: S=";
    printOpOneLine(forkOp);
    llvm::outs() << "  E=";
    printOpOneLine(joinOp);
    llvm::outs() << "\n";

    return subgraph(forkOp, joinOp);
  }

  subgraph matchMergeShrinkPattern(Operation *peakOp) {

    std::queue<Operation *> findConcatWorklist;
    llvm::DenseSet<Operation *> findConcatVisited;
    Operation *concatOp = nullptr;

    std::queue<Operation *> findShrinkerWorklist;
    llvm::DenseSet<Operation *> findShrinkerVisited;
    Operation *shrinker = nullptr;

    findConcatWorklist.push(peakOp);
    findShrinkerWorklist.push(peakOp);

    auto findConcatOpBackward = [&]() -> Operation * {
      while (!findConcatWorklist.empty()) {
        Operation *currentOp = findConcatWorklist.front();
        findConcatWorklist.pop();
        findConcatVisited.insert(currentOp);

        for (Value v : currentOp->getOperands()) {
          Operation *defOp = v.getDefiningOp();
          // defOp가 null이거나 Constant거나 noValue면 스킵
          // 이미 방문한 노드도 스킵
          if (findConcatVisited.contains(defOp) || !defOp ||
              isa<ONNXConstantOp>(defOp) || isa<ONNXNoneOp>(defOp))
            continue;
          findConcatWorklist.push(defOp);
        }

        if (isa<ONNXConcatOp>(currentOp)) {
          return currentOp;
        }
      }

      return nullptr;
    };

    auto findShrinkerForward = [&]() -> Operation * {
      while (!findShrinkerWorklist.empty()) {
        Operation *currentOp = findShrinkerWorklist.front();
        findShrinkerWorklist.pop();
        findShrinkerVisited.insert(currentOp);

        for (Value v : currentOp->getResults()) {
          for (Operation *userOp : v.getUsers()) {
            // 이미 방문한 노드는 스킵
            if (findShrinkerVisited.contains(userOp))
              continue;
            findShrinkerWorklist.push(userOp);
          }
        }

        if (checkTensorSizeChange(currentOp) == TensorSizeChange::SHRINK) {
          return currentOp;
        }
      }
      // 못찾음
      return nullptr;
    };

    if (!(concatOp = findConcatOpBackward())) {
      llvm::outs() << "[match Merge/Shrink] Cannot find Concat Op \n";
      return subgraph{};
    } else {
      llvm::outs() << "[match Merge/Shrink] found Concat Op: ";
      printOpOneLine(concatOp);
      llvm::outs() << '\n';
    }

    if (!(shrinker = findShrinkerForward())) {
      llvm::outs() << "[match Merge/Shrink] Cannot find Shrinker \n";
      return subgraph{};
    } else {
      llvm::outs() << "[match Merge/Shrink] found Shrinker: ";
      printOpOneLine(shrinker);
      llvm::outs() << '\n';
    }

    IsolateStatus status = checkIsolation(subgraph(concatOp, shrinker));

    while (status != IsolateStatus::Isolated) {
      if (status == IsolateStatus::ExternalInput) {
        if (!(concatOp = findConcatOpBackward())) {
          llvm::outs() << "[match Merge/Shrink] Cannot find Concat Op \n";
          return subgraph{};
        } else {
          llvm::outs() << "[match Merge/Shrink] found Concat Op: ";
          printOpOneLine(concatOp);
          llvm::outs() << '\n';
          status = checkIsolation(subgraph(concatOp, shrinker));
        }
      } else if (status == IsolateStatus::ExternalOutput) {
        if (!(shrinker = findShrinkerForward())) {
          llvm::outs() << "[match Merge/Shrink] Cannot find Shrinker \n";
          return subgraph{};
        } else {
          llvm::outs() << "[match Merge/Shrink] found Shrinker: ";
          printOpOneLine(shrinker);
          llvm::outs() << '\n';
          status = checkIsolation(subgraph(concatOp, shrinker));
        }
      } else if (status == IsolateStatus::NotSplittable) {
        llvm::outs() << "[match Merge/Shrink] status == NotSplittable \n";
        return subgraph{};
      }
    }

    return subgraph(concatOp, shrinker);
  }

  /*=========opimize logic=========*/
  static bool isDimPreserved(
      Operation *op, size_t dimension, const subgraph &subgraph) {
    const Operation *s = subgraph.getS();
    const Operation *e = subgraph.getE();

    if (op == e) {
      return true;
    }
    // Conv일 경우
    if (auto convOp = dyn_cast<ONNXConvOp>(op)) {
      // channel dimension일 경우
      if (dimension == 1) {
        int64_t group = convOp.getGroup();
        Value input = convOp.getX();
        auto tensorType = dyn_cast<TensorType>(input.getType());
        ArrayRef<int64_t> shape = tensorType.getShape();
        int64_t inChannel = shape[1];

        // convOp가 depthwise convolution이 아니라면
        if (group != inChannel) {
          llvm::outs() << "[isDimPreserved] Failed: Check channel dimension "
                          "dependency but convolution is not Depthwis conv. \n";
          return false;
        }
      }

      Value outVal = convOp.getResult();
      for (Operation *userOp : outVal.getUsers()) {
        return isDimPreserved(userOp, dimension, subgraph);
      }
    }
    // MatMul일 경우
    else if (auto matmulOp = dyn_cast<ONNXMatMulOp>(op)) {
      Value a = matmulOp.getA();
      Value b = matmulOp.getB();
      Operation *opA = a.getDefiningOp();
      Operation *opB = b.getDefiningOp();

      if (!isa<ONNXConstantOp>(opA)) {
        TensorType tensorTypeA = dyn_cast<TensorType>(a.getType());
        auto shape = tensorTypeA.getShape();
        // 뒤에서 첫번째 차원
        if (shape.size() - 1 == dimension) {
          llvm::outs() << "[isDimPreserved] Failed: target dimension is "
                          "reduction dimension. \n  Op: ";
          printOpOneLine(op);
          llvm::outs() << "\n Dim: " << dimension;
          return false;
        }
      }

      if (!isa<ONNXConstantOp>(opB)) {
        TensorType tensorTypeB = dyn_cast<TensorType>(b.getType());
        auto shape = tensorTypeB.getShape();
        // 뒤에서 두번째 차원
        if (shape.size() - 2 == dimension) {
          llvm::outs() << "[isDimPreserved] Failed: target dimension is "
                          "reduction dimension. Op: ";
          printOpOneLine(op);
          llvm::outs() << "\n";
          return false;
        }
      }

      Value outVal = matmulOp.getResult();
      for (Operation *userOp : outVal.getUsers()) {
        return isDimPreserved(userOp, dimension, subgraph);
      }
    }
    // Reshape일 경우
    else if (auto reshapeOp = dyn_cast<ONNXReshapeOp>(op)) {
      Value inVal = reshapeOp.getData();
      TensorType inputTensorType = dyn_cast<TensorType>(inVal.getType());
      auto inputShape = inputTensorType.getShape();

      Value outVal = reshapeOp.getReshaped();
      TensorType outputTensorType = dyn_cast<TensorType>(outVal.getType());
      auto outputShape = outputTensorType.getShape();

      int64_t targetDimSize = inputShape[dimension];

      int64_t inputProd = 1;
      for (size_t i = 0; i < dimension; i++) {
        inputProd = inputProd * inputShape[i];
      }

      // target dimension이 보존되는지 검증
      int64_t outputProd = 1;
      for (size_t i = 0; i < outputShape.size(); i++) {
        if (outputShape[i] == targetDimSize) {
          for (size_t j = 0; j < i; j++) {
            outputProd = outputProd * outputShape[j];
          }
          if (inputProd == outputProd) {
            Value outVal = reshapeOp.getResult();
            for (Operation *userOp : outVal.getUsers()) {
              return isDimPreserved(
                  userOp, i, subgraph); // i == 바뀐 target dimension
            }
          }
        }
      }
      llvm::outs() << "[isDimPreserved] Fail: ReshapeOp fuses or splits target "
                      "dimension. \n    Op: ";
      printOpOneLine(reshapeOp);
      llvm::outs() << "\n    Dimension: " << dimension << "\n";
      return false;
    }
    // concatOp일 경우
    else if (auto concatOp = dyn_cast<ONNXConcatOp>(op)) {
      if (s == concatOp) {
        Value outVal = concatOp.getConcatResult();
        for (Operation *userOp : outVal.getUsers()) {
          return isDimPreserved(userOp, dimension, subgraph);
        }
      } else {
        llvm::outs() << "[isDimPreserved] failed: Unexpected ConcatOp.\n";
      }
    }
    // MaxPool
    else if (isa<ONNXMaxPoolSingleOutOp>(op)) {
      if (dimension != 1) {
        llvm::outs()
            << "[isDimPreserved] fail: MaxPool has spatial dependency. Only "
               "Channel dim (1) is splittable. Current dim: "
            << dimension << "\n";
        return false;
      }
      Value outVal = op->getResult(0);
      for (Operation *userOp : outVal.getUsers()) {
        return isDimPreserved(userOp, dimension, subgraph);
      }
    }
    // elementwise 연산일 경우
    else if (isa<ONNXAddOp, ONNXClipOp, ONNXMulOp, ONNXSigmoidOp, ONNXGeluOp,
                 ONNXReluOp>(op)) {
      if (op == e) {
        return true;
      }
      if (op->getNumResults() != 1) {
        llvm::outs() << "[isDimPreserved] fail: Multiple results at Op: ";
        printOpOneLine(op);
        llvm::outs() << "\n";
        return false;
      }
      Value outVal = op->getResult(0);
      for (Operation *userOp : outVal.getUsers()) {
        return isDimPreserved(userOp, dimension, subgraph);
      }
    }
    llvm::outs() << "[isDimPreserved] Failed: Unknown Op: ";
    printOpOneLine(op);
    llvm::outs() << "\n";
    return false;
  }

  llvm::DenseSet<int64_t> getSplittableDims(subgraph &subgraph) {
    llvm::DenseSet<int64_t> splittableDims;

    if (subgraph.getS()->getNumResults() != 1) {
      llvm::outs() << "[getSplittableDims] Fail: s has multiple results. \n";
    }

    Value sOut = subgraph.getS()->getResult(0);
    TensorType tensorType = dyn_cast<RankedTensorType>(sOut.getType());
    int64_t rank = tensorType.getRank();
    llvm::outs() << "[getSplittableDims] Rank: " << rank << "\n";

    auto userOps = sOut.getUsers();

    for (int dimension = 1; dimension < rank; dimension++) {
      bool preserved = false;

      for (Operation *op : userOps) {
        preserved = isDimPreserved(op, dimension, subgraph);
        if (!preserved) {
          break;
        }
      }
      if (preserved) {
        splittableDims.insert(dimension);
      }
    }

    llvm::outs() << "[getSplittableDims] Splittable Dimensions: ";
    for (int64_t dim : splittableDims) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "\n";
    return splittableDims;
  }

  // TODO:
  int64_t determineSplitAxis(subgraph &subgraph) {
    if (subgraph.getS()->getNumResults() != 1) {
      llvm::outs() << "[determineSplitAxis] Fail: s has multiple results. \n";
    }

    Value sOutput = subgraph.getS()->getResult(0);
    auto tensorType = dyn_cast<TensorType>(sOutput.getType());
    ArrayRef<int64_t> shape = tensorType.getShape();

    // 1: Channel dimension (typically dim 1 for NCHW)
    // int64_t dimension1 = shape[1];
    // s가 무슨 연산인지에 따라
    if (auto matmulOp = dyn_cast<ONNXMatMulOp>(subgraph.getS())) {
      // TODO: MatMul 처리 로직

    } else if (auto convOp = dyn_cast<ONNXConvOp>(subgraph.getS())) {
      // TODO: Conv 처리 로직

      // isa는 여러 타입을 동시에 확인 가능
    } else if (isa<ONNXAddOp, ONNXClipOp>(subgraph.getS())) {
      // TODO: Add 또는 Clip 처리 로직

    } else {
      // 그 외 다른 Op들
    }

    // Priority 2: Spatial dimensions (last dim for flattened tensors)
    for (int64_t i = shape.size() - 1; i >= 0; i--) {
      if (shape[i] > 1 && shape[i] % 2 == 0) {
        return i;
      }
    }

    // Priority 3: Batch dimension (fallback)
    return 0;
  }

  static SmallVector<Value, 4> splitValue(OpBuilder &builder, Location loc,
      Value input, int64_t axis = 0, int64_t numOutputs = 2) {
    SmallVector<Value, 4> results;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(input);

    if (numOutputs <= 0) {
      llvm::outs() << "[splitValue] fail: numOutputs must be > 0.\n";
      return results;
    }

    if (!input || isa<NoneType>(input.getType())) {
      Value none = builder.create<ONNXNoneOp>(loc).getResult();
      for (int64_t i = 0; i < numOutputs; ++i)
        results.push_back(none);
      return results;
    }

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() <= axis) {
      llvm::outs()
          << "[splitValue] fail: Input value is unranked or (rank < axis). \n";
      return results;
    }

    auto shape = inputType.getShape();
    auto elementTy = inputType.getElementType();
    int64_t totalSize = shape[axis];

    // 1. 각 분할의 크기 계산
    int64_t baseSize = totalSize / numOutputs;
    int64_t remainder = totalSize % numOutputs;

    SmallVector<int64_t, 4> splitSizes;
    splitSizes.reserve(numOutputs);

    // 나머지를 앞쪽 인덱스에 1씩 나누어 줌 (예: 10/3 -> 4, 3, 3)
    for (int64_t i = 0; i < numOutputs; ++i) {
      int64_t size = baseSize + (i < remainder ? 1 : 0);
      splitSizes.push_back(size);
    }

    // 2. 각 output의 타입(Shape) 계산
    SmallVector<Type, 4> outTypes;
    outTypes.reserve(numOutputs);
    for (int64_t size : splitSizes) {
      SmallVector<int64_t, 4> outShape(shape.begin(), shape.end());
      outShape[axis] = size;
      outTypes.push_back(RankedTensorType::get(outShape, elementTy));
    }

    auto noneVal = builder.create<ONNXNoneOp>(loc).getResult();
    auto axisAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, /*isSigned=*/true), axis);
    auto numOutputsAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, /*isSigned=*/true), numOutputs);

    // 3. 'split' 입력 값 생성 (균등 분할이 아닌 경우 필수)
    Value splitVal = noneVal;
    if (remainder != 0) {
      // split 정보를 담은 Constant Op 생성 (1D Tensor)
      auto splitType =
          RankedTensorType::get({numOutputs}, builder.getI64Type());
      auto splitAttr =
          DenseElementsAttr::get(splitType, llvm::ArrayRef(splitSizes));

      // ONNXConstantOp 생성 (value attribute에 값 할당)
      splitVal = builder.create<ONNXConstantOp>(loc, Attribute(), splitAttr)
                     .getResult();
    }

    // 4. ONNXSplitOp 생성
    // splitVal이 None이면 균등 분할, 아니면 splitVal에 명시된
    // 대로 분할
    auto splitOp = builder.create<ONNXSplitOp>(
        loc, TypeRange(outTypes), input, splitVal, axisAttr, numOutputsAttr);

    for (int64_t i = 0; i < numOutputs; ++i)
      results.push_back(splitOp.getResult(i));

    return results;
  }

  // ConvOp를 받아서 타입을 문자열로 반환하는 함수
  std::string getConvTypeName(ONNXConvOp op) {
    // 1. Group 속성 확인
    int64_t group = op.getGroup();

    // 2. 입력 채널 수 확인
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    if (!inputType)
      return "Unknown (Unranked Input)";

    int64_t inputChannels = inputType.getShape()[1]; // NCHW 기준

    // 3. 커널 크기가 1x1인지 확인
    auto weightType = dyn_cast<RankedTensorType>(op.getW().getType());
    if (!weightType)
      return "Unknown (Unranked Weight)";

    llvm::ArrayRef<int64_t> weightShape = weightType.getShape();
    bool is1x1 = true;

    // ONNX Weight: [M, C/group, kH, kW] -> 인덱스 2부터 공간 차원
    for (size_t i = 2; i < weightShape.size(); ++i) {
      if (weightShape[i] != 1) {
        is1x1 = false;
        break;
      }
    }

    // --- 판별 로직 ---

    // Case 1: Depthwise Convolution
    // (입력 채널이 1보다 크고, 그룹 수와 채널 수가 같은 경우)
    if (group == inputChannels && inputChannels > 1) {
      return "Depthwise";
    }

    // Case 2: Pointwise Convolution
    // (1x1 커널이면서 그룹이 1인 경우)
    if (is1x1 && group == 1) {
      return "Pointwise";
    }

    // Case 3: Grouped Convolution
    // (Depthwise는 아니지만 그룹이 나뉘어 있는 경우)
    if (group > 1 && group < inputChannels) {
      return "Grouped";
    }

    // Case 4: Standard Convolution
    // (그 외 일반적인 경우)
    return "Standard";
  }

  void splitSubgraph(subgraph subgraph, int64_t splitDim) {
    llvm::outs() << "[splitSubgraph] Splitting subgraph at dimension "
                 << splitDim << "\n";

    enum class MergeKind { Concat, Add };
    MergeKind mergeKind = MergeKind::Concat;

    Operation *s = subgraph.getS();
    Operation *e = subgraph.getE();
    OpBuilder builder(s);
    Location loc = s->getLoc();
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(builder, loc);

    Value Y1, Y2;

    /* 1. Split S Node (Start Node)
    // S == ConvOp                           */
    if (auto convOp = dyn_cast<ONNXConvOp>(s)) {
      Value x = convOp.getX();
      Value w = convOp.getW();
      Value b = convOp.getB();
      StringAttr autoPad = convOp.getAutoPadAttr();
      ArrayAttr dilations = convOp.getDilationsAttr();
      IntegerAttr group = convOp.getGroupAttr();
      ArrayAttr kernelShape = convOp.getKernelShapeAttr();
      ArrayAttr pads = convOp.getPadsAttr();
      ArrayAttr strides = convOp.getStridesAttr();

      auto makeConv = [&](Value x, Value w, Value b,
                          IntegerAttr group) -> Value {
        return builder
            .create<ONNXConvOp>(loc, x, w, b, autoPad, dilations, group,
                kernelShape, pads, strides)
            .getResult();
      };

      if (splitDim == 1) {
        auto wSplits = splitValue(builder, loc, w, 0);
        auto bSplits = splitValue(builder, loc, b, 0);
        Y1 = makeConv(x, wSplits[0], bSplits[0], group);
        Y2 = makeConv(x, wSplits[1], bSplits[1], group);
      } else {
        // Handle other dimensions or error
        llvm::outs()
            << "[splitSubgraph] fail: Unsupported dimension for Conv.\n";
        return;
      }
    }
    // S == MatmulOp
    else if (auto matmulOp = dyn_cast<ONNXMatMulOp>(s)) {
      Value A = matmulOp.getA();
      Value B = matmulOp.getB();
      auto aTy = dyn_cast<RankedTensorType>(A.getType());
      auto bTy = dyn_cast<RankedTensorType>(B.getType());
      auto sOutRank =
          dyn_cast<RankedTensorType>(matmulOp.getY().getType()).getRank();
      auto unrankedOutTy = UnrankedTensorType::get(aTy.getElementType());

      auto makeMatMul = [&](Value lhs, Value rhs) -> Value {
        return builder.create<ONNXMatMulOp>(loc, unrankedOutTy, lhs, rhs)
            .getResult();
      };

      // (S출력의 rank - splitDim > 2) -> splitDim이 broadcasting되는 차원임
      if (sOutRank - splitDim > 2) {
        auto aSplits = splitValue(builder, loc, A, splitDim);
        Y1 = makeMatMul(aSplits[0], B);
        Y2 = makeMatMul(aSplits[1], B);
      }
      // splitDim이 MxK @ KxN 에서 M일때
      else if (sOutRank - splitDim == 2) {
        auto aSplits = splitValue(builder, loc, A, aTy.getRank() - 2);
        Y1 = makeMatMul(aSplits[0], B);
        Y2 = makeMatMul(aSplits[1], B);
      }
      // splitDim이 MxK @ KxN 에서 N일때
      else if (sOutRank - splitDim == 1) {
        auto bSplits = splitValue(builder, loc, B, bTy.getRank() - 1);
        Y1 = makeMatMul(A, bSplits[0]);
        Y2 = makeMatMul(A, bSplits[1]);
      }

    } else {
      llvm::outs() << "[splitSubgraph] fail: Unknown Operation at S.\n";
      return;
    }

    /* 2. Common Path Processing Lambda                         */
    // pathIdx: 0 for Path A, 1 for Path B
    auto processPath = [&](Value inputVal, int pathIdx) -> Value {
      IRMapping mapping;
      Value sOut = s->getResult(0);
      mapping.map(sOut, inputVal);

      builder.setInsertionPointAfterValue(inputVal);

      // 첫 번째 노드(s)를 제외한 나머지 노드 복제 및 수정
      for (Operation *op : llvm::drop_begin(subgraph.subgraphNodes, 1)) {
        Operation *clonedOp = builder.clone(*op, mapping);

        // 2-1. ConvOp 처리
        // TODO:
        // splitDim이 reduction일 경우 mergeKind = MergeKind::Add로 바꿔주기
        if (auto convOp = dyn_cast<ONNXConvOp>(clonedOp)) {
          if (splitDim == 1) {
            // Weight & Bias Splitting
            Value w = convOp.getW();
            Value b = convOp.getB();

            // splitValue 호출 후 pathIdx에 해당하는 조각 선택
            auto wSplit = splitValue(builder, convOp.getLoc(), w, 0)[pathIdx];
            auto bSplit = splitValue(builder, convOp.getLoc(), b, 0)[pathIdx];

            convOp.setOperand(1, wSplit);
            convOp.setOperand(2, bSplit);

            // Group Attribute Update
            auto si64Ty =
                IntegerType::get(builder.getContext(), 64, IntegerType::Signed);
            auto newGroupAttr = IntegerAttr::get(
                si64Ty, dyn_cast<TensorType>(wSplit.getType()).getShape()[0]);
            convOp.setGroupAttr(newGroupAttr);
          }
        }

        // 2-2. MatmulOp 처리
        else if (auto matmulOp = dyn_cast<ONNXMatMulOp>(clonedOp)) {
          // 원본 op에서 타입 정보 가져오기 (clonedOp는 이미 Unranked로 변경됨)
          auto origMatmulOp = dyn_cast<ONNXMatMulOp>(op);
          Value origB = origMatmulOp.getB();
          auto bTy = dyn_cast<RankedTensorType>(origB.getType());

          // clonedOp에서 현재 operand 가져오기
          Value B = matmulOp.getB();
          Operation *bDefOp = B.getDefiningOp();
          bool bIsConst = bDefOp && isa<ONNXConstantOp>(bDefOp);

          // splitDim은 현재 연산으로 들어오는 activation의 split dimension을
          // 의미함. matmul에서 가능한 경우는
          // activation X weight, activation X activation
          // 두 가지인것으로 보여짐. 따라서 A는 일단 activation이라고 가정하자.
          if (splitDim == 1) {
            // splitDim == 1: A의 마지막에서 두 번째 차원(M)으로 분할
            // A는 mapping으로 이미 분할된 값이 들어올거고 B는 분할 할 필요 없음
          } else if (splitDim == 2) {
            // splitDim == 2: A의 마지막 차원(K)으로 분할
            // ** op == e인 경우에만 여기 들어올 수 있음

            // e에서 splitDim이 reduction되는 차원이므로 마지막에 Add로
            // merge해야함.
            mergeKind = MergeKind::Add;

            // B가 상수(weight)인 경우 분할 필요
            if (bIsConst) {
              int64_t splitAxis = bTy.getRank() - 2;
              auto bSplit =
                  splitValue(builder, matmulOp.getLoc(), B, splitAxis)[pathIdx];
              matmulOp.setOperand(1, bSplit);
            }
            // B가 변수라면 이미 mapping을 통해 분할된 값이 전달됨
          } else {
            llvm::outs() << "[splitSubgraph] Warning: Unsupported splitDim for "
                            "MatMul in path.\n";
          }
        }

        // 2-3. ReshapeOp 처리
        else if (auto reshapeOp = dyn_cast<ONNXReshapeOp>(clonedOp)) {
          Value shapeVal = reshapeOp.getShape();
          if (auto constOp =
                  dyn_cast<ONNXConstantOp>(shapeVal.getDefiningOp())) {
            if (auto denseAttr =
                    dyn_cast<DenseElementsAttr>(constOp.getValueAttr())) {
              builder.setInsertionPoint(constOp); // 상수 생성 위치 설정

              SmallVector<int64_t, 4> shapeVals;
              for (int64_t v : denseAttr.getValues<int64_t>())
                shapeVals.push_back(v);

              if (shapeVals[splitDim] % 2 == 0) {
                shapeVals[splitDim] /= 2; // 차원 반으로 줄임
                Value newConstVal = create.onnx.constantInt64(
                    llvm::ArrayRef<int64_t>(shapeVals));
                reshapeOp.setOperand(1, newConstVal);
              } else {
                llvm::errs()
                    << "[splitSubgraph] Warning: dim not divisible by 2.\n";
              }
              // 원복
              builder.setInsertionPoint(clonedOp);
            }
          }
        }
        // 2-4. Binary Element-wise (Add, Mul) 처리
        else if (isa<ONNXAddOp, ONNXMulOp>(clonedOp)) {
          bool hasConst = false;

          for (Value v : op->getOperands()) {
            Operation *defOp = v.getDefiningOp();
            if (isa<ONNXConstantOp>(defOp))
              hasConst = true;
          }

          if (hasConst) {
            int64_t constRank, actRank;
            Value constVal;

            // cloned subgraph에서는 clonedOp의 상위 연산이 수정된 상태임.
            // 따라서 원본 op에서 rank를 구해야함.
            for (Value v : op->getOperands()) {
              Operation *defOp = v.getDefiningOp();
              auto rtt = dyn_cast<RankedTensorType>(v.getType());

              // operand가 Constant일때
              if (isa<ONNXConstantOp>(defOp)) {
                constRank = rtt.getRank();
                constVal = v;
              }
              // operand가 activation 일때
              else {
                actRank = rtt.getRank();
              }
            }

            uint64_t operandIdx = 0;
            int64_t constSplitDim = splitDim - (actRank - constRank);

            auto rtt = dyn_cast<RankedTensorType>(constVal.getType());

            // clonedOp의 operand를 순회하여 constant를 찾아 분할
            for (Value v : clonedOp->getOperands()) {
              Operation *defOp = v.getDefiningOp();
              // Constant이면서 Broadcasting이 아닌 경우(splitDimSize > 1)에만
              // 분할
              if (isa<ONNXConstantOp>(defOp) && constSplitDim > -1) {
                int64_t splitDimSize = rtt.getDimSize(constSplitDim);
                if (splitDimSize > 1) {
                  auto splitVal =
                      splitValue(builder, loc, v, constSplitDim)[pathIdx];
                  clonedOp->setOperand(operandIdx, splitVal);
                } else {
                  llvm::outs() << "[subgraphSplit] fail: constSplitDim < 0";
                }
              }
              operandIdx++;
            }
          }
        }

        // Unary Element-wise는 아무것도 안해줘도 됨
        // ex. clip, gelu, sigmoid

        // 결과 타입 갱신 (Unranked로 변경하여 추후 Shape Inference 유도)
        Value y = clonedOp->getResult(0);
        y.setType(UnrankedTensorType::get(
            dyn_cast<TensorType>(y.getType()).getElementType()));

        builder.setInsertionPointAfter(clonedOp);
      }

      // 경로의 마지막 노드의 결과값 반환
      return mapping.lookup(e->getResult(0));
    };

    /*==========================================================*/
    /* 3. Execute Paths & Merge (Concat or Add)                 */
    /*==========================================================*/

    // Path A 생성
    Value outA = processPath(Y1, 0);

    // Path B 생성
    Value outB = processPath(Y2, 1);

    // Insertion Point 설정 (마지막 연산 뒤)
    builder.setInsertionPointAfterValue(outB);

    // 공통적으로 사용할 타입 정의
    auto elemTy = dyn_cast<TensorType>(outA.getType()).getElementType();
    auto unrankedOutTy = UnrankedTensorType::get(elemTy);

    Value mergedResult;

    if (mergeKind == MergeKind::Concat) {
      IntegerAttr concatAxisAttr =
          builder.getIntegerAttr(builder.getIntegerType(64, true), splitDim);

      auto concatOp = builder.create<ONNXConcatOp>(
          s->getLoc(), unrankedOutTy, ValueRange{outA, outB}, concatAxisAttr);

      mergedResult = concatOp.getResult();

    } else if (mergeKind == MergeKind::Add) {
      auto addOp =
          builder.create<ONNXAddOp>(s->getLoc(), unrankedOutTy, outA, outB);

      mergedResult = addOp.getResult();
    }

    // 결과 교체
    if (e->getNumResults() == 1) {
      e->getResult(0).replaceAllUsesWith(mergedResult);
    }

    /*==========================================================*/
    /* 4. Cleanup                                               */
    /*==========================================================*/
    for (auto it = subgraph.subgraphNodes.rbegin();
         it != subgraph.subgraphNodes.rend(); it++) {
      (*it)->erase();
    }
    subgraph.subgraphNodes.clear();

    llvm::outs() << "[splitSubgraph] Successfully optimized.\n";
  }

  /*-------------------------------------------*/

  void optConcat(subgraph mergeShrinkSubgraph) {
    llvm::outs() << "[optConcat] Optimizing Merge/Shrink pattern.\n";

    Operation *s = mergeShrinkSubgraph.getS();
    Operation *e = mergeShrinkSubgraph.getE();

    auto concatOp = dyn_cast<ONNXConcatOp>(s);
    if (!concatOp) {
      llvm::outs() << "[optConcat] fail: S node must be concatOp.\n";
    }

    int64_t splitDim = concatOp.getAxis();

    OpBuilder builder(s);
    Location loc = s->getLoc();
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(builder, loc);

    Value concatResult = concatOp.getResult();

    SmallVector<Value, 4> branchOutputs;

    // 각 브랜치에서 새로 만든 마지막 op를 추적
    Operation *prev = s;

    // concat의 각 입력에 대해 브랜치 하나씩 만들기
    auto inputs = concatOp.getInputs();
    for (size_t i = 0; i < inputs.size(); i++) { // 또는 s->getOperands()
      Value in = inputs[i];
      IRMapping mapping;

      // concat 결과를 이 브랜치의 입력으로 치환
      mapping.map(concatResult, in);

      // subgraphNodes: [S, op1, op2, ..., E] 라고 가정하고
      for (Operation *op :
          llvm::drop_begin(mergeShrinkSubgraph.subgraphNodes, 1)) {
        // 원래 op 바로 뒤에 두는 대신, 이 브랜치의 prev 뒤에 붙이는 게 깔끔함
        builder.setInsertionPointAfter(prev);

        // 핵심 2: clone(*op, mapping) 사용
        Operation *clonedOp = builder.clone(*op, mapping);

        if (op != e) {
          if (auto convOp = dyn_cast<ONNXConvOp>(clonedOp)) {
            std::string convType = getConvTypeName(convOp);
            if (splitDim == 1) {
              if (convType == "Depthwise") {
                // TODO:
                //   - group attr 수정 -> in_channel로
                //   - weight, bias 쪼개기
              } else if (convType == "Grouped") {
                llvm::outs() << "[optConv] fail: Unexpected grouped conv.\n";
              }
              // depthwise일 경우에만 split 가능
              else {
                llvm::outs() << "[optConv] fail: splitDim == 1 but conv is not "
                                "depthwise.\n";
                return;
              }
            } else if (splitDim == 2 || splitDim == 3) {
              // TODO:
            } else {
              llvm::outs() << "[optConcat] fail: Unexpected splitDim: "
                           << splitDim << "\n";
            }
          }
        } else if (op == e) {
          if (auto convOp = dyn_cast<ONNXConvOp>(clonedOp)) {
            std::string convType = getConvTypeName(convOp);

            if (splitDim == 1) {
              if (convType == "Standard" || convType == "Pointwise") {
                Value weight = convOp.getW();

                // weight의 idx 1번 차원이 in_channels/group임
                // 일단은 group 1이라고 가정함
                // TODO: group이 1이 아닐경우 고려해서 수정
                builder.setInsertionPoint(convOp);
                Value noneValue = builder.create<ONNXNoneOp>(convOp.getLoc());
                auto splits =
                    splitValue(builder, loc, weight, 1, inputs.size());
                convOp.setOperand(1, splits[i]);
                if (i != 0) {
                  convOp.setOperand(2, noneValue);
                }
                builder.setInsertionPointAfter(convOp);
              } else if (convType == "Depthwise") {
                // TODO:
                llvm::outs() << "[optConcat] Depthwise convolution \n";
              } else if (convType == "Grouped") {
                llvm::outs() << "[optConcat] fail: Unexpected grouped conv "
                                "at E node.\n";
              }
            } else if (splitDim == 2 || splitDim == 3) {
              // TODO:
              //  height, width로 쪼갤때
            } else {
              llvm::outs() << "[optConcat] fail: Unexpected splitDim: "
                           << splitDim << "\n";
            }
          }
        }
        // 결과 타입 수정
        Value y = clonedOp->getResult(0);
        auto elemType = dyn_cast<TensorType>(y.getType()).getElementType();
        auto newType = UnrankedTensorType::get(elemType);
        y.setType(newType);

        prev = clonedOp;
      }

      // E에 해당하는 op가 마지막에 클론되었을 테니까
      // 그 결과를 나중에 다시 합치거나 쓸 수 있음
      branchOutputs.push_back(prev->getResult(0));
    }

    // branchOutputs 합치기
    // 현재 E가 standard conv일 경우에 대해서면 구현 (Add로 합침)
    // TODO:
    //  - 나머지 conv에 대해서는 concat으로 합치고
    //  - MatMul일 경우에는 splitDim이 reduction axis라면 sum, 아니면 concat으로
    //    합치면 됨
    if (auto convOp = dyn_cast<ONNXConvOp>(e)) {
      std::string convType = getConvTypeName(convOp);
      if (splitDim == 1) {
        if (convType == "Standard" || convType == "Pointwise") {
          Value y = convOp.getY();
          auto elemType = dyn_cast<TensorType>(y.getType()).getElementType();
          auto newType = UnrankedTensorType::get(elemType);
          Value sumResult = create.onnx.sum(newType, branchOutputs);

          // e의 result를 sum의 result로 갈아끼우기
          y.replaceAllUsesWith(sumResult);
        }
      }
    }

    /*------------기존 연산 지우기------------*/
    // subgraph의 연산들에 역순으로 접근
    for (auto it = mergeShrinkSubgraph.subgraphNodes.rbegin(),
              end = mergeShrinkSubgraph.subgraphNodes.rend();
         it != end; it++) {
      Operation *op = *it;
      op->erase();
    }
    mergeShrinkSubgraph.subgraphNodes.clear();

    llvm::outs()
        << "[optConcat] Merge/Shrink pattern is successfully optimized.\n";
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
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createPeakMemOptPass() {
  return std::make_unique<PeakMemOptPass>();
}
} // namespace onnx_mlir
