#include <algorithm>
#include <deque>
#include <queue>
#include <system_error>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
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
    // auto &domInfo = getAnalysis<DominanceInfo>();
    // auto &postDomInfo = getAnalysis<PostDominanceInfo>();

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

    /* ---------------Expand/Shrink--------------- */
    const uint32_t DETECTION_SCOPE = 64;

    llvm::outs() << "[match] try EXPAND~SHRINK from peak, scope="
                 << DETECTION_SCOPE << "\n";

    subgraph expandShrinkSubgraph =
        matchExpandShrinkPattern(peakOp, DETECTION_SCOPE);

    if (!expandShrinkSubgraph.s) {
      llvm::outs() << "[match Expand/Shrink] no window\n";
    } else {
      llvm::outs() << "[match Expand/Shrink] SPLITTABLE window: S=";
      printOpOneLine(expandShrinkSubgraph.s);
      llvm::outs() << "  E=";
      printOpOneLine(expandShrinkSubgraph.e);
      llvm::outs() << "\n";
    }
    /* ---------------Expand/Shrink--------------- */

    /* ---------------Fork/Join--------------- */
    IsolatedSubgraph isolatedSubgraph;

    if (matchForkJoinPattern(peakOp, funcOp, isolatedSubgraph)) {
      llvm::outs() << "[match Fork/Merge] SPLITTABLE window: S=";
      printOpOneLine(isolatedSubgraph.entryGate);
      llvm::outs() << "  E=";
      printOpOneLine(isolatedSubgraph.exitGate);
      llvm::outs() << "\n";
    } else {
      llvm::outs() << "[match Fork/Merge] no window\n";
    }
    /* ---------------Fork/Join--------------- */

    /* ---------------Merge/Shrink--------------- */
    subgraph mergeShrinkSubgraph = matchMergeShrinkPattern(peakOp);
    if (mergeShrinkSubgraph.s) {
      llvm::outs() << "[match Merge/Shrink] SPLITTABLE window: S=";
      printOpOneLine(mergeShrinkSubgraph.s);
      llvm::outs() << "  E=";
      printOpOneLine(mergeShrinkSubgraph.e);
      llvm::outs() << "\n";
    } else {
      llvm::outs() << "[match Merge/Shrink] no window\n";
    }
    /* ---------------Merge/Shrink--------------- */

    /* ===============Split=============== */
    if (expandShrinkSubgraph.s) {
      llvm::DenseSet<int64_t> splittableDims =
          getSplittableDims(expandShrinkSubgraph);

      llvm::outs() << "[Splittable Dims] ";
      for (int64_t dim : splittableDims) {
        llvm::outs() << dim << ' ';
      }
      llvm::outs() << '\n';
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
    Isolated,
    ExternalInput,
    ExternalOutput,
    NotSplittable
  };

  struct subgraph {
    Operation *s = nullptr; // start node
    Operation *e = nullptr; // End node
    llvm::DenseSet<Operation *>
        subgraphNodes;           // All ops in s→e path including s and e
    int64_t splitDimension = -1; // Dimension to split along
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

    llvm::outs() << "[checkTensorSizeChange] ";
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

      for (Value result : currentOp->getResults()) {
        for (Operation *userOp : result.getUsers()) {
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

  // s에서 시작해 e로 끝나는 서브 그래프 노드 집합 반환
  static llvm::DenseSet<Operation *> getSubgraphNodes(
      Operation *s, Operation *e) {
    if (!s) {
      llvm::outs() << "[getSubgraphNodes] s = nullptr \n";
      return {};
    }
    if (!e) {
      llvm::outs() << "[getSubgraphNodes] e = nullptr \n";
      return {};
    }

    llvm::DenseSet<mlir::Operation *> forwardSet =
        getForwardReachableNodes(s, e);

    llvm::DenseSet<mlir::Operation *> backwardSet =
        getBackwardReachableNodes(s, e);

    llvm::DenseSet<mlir::Operation *> subgraphNodes;
    for (mlir::Operation *op : forwardSet) {
      if (backwardSet.contains(op)) {
        subgraphNodes.insert(op);
      }
    }

    return subgraphNodes;
  }

  // 서브그래프 고립성 검사
  static IsolateStatus isIsolatedSubgraph(Operation *s, Operation *e) {
    llvm::DenseSet<Operation *> subgraphNodes = getSubgraphNodes(s, e);

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
            llvm::outs()
                << "[isIsolatedSubgraph] Failed: External input. Node ("
                << op->getName() << ") is not S, but it receives "
                << "a BlockArgument (" << operand << ") as input.\n";
            return IsolateStatus::NotSplittable;
          }

          if (!subgraphNodes.contains(defOp)) {
            // Case 2: Op Result (S 또는 서브그래프 내부 노드가 아닌 곳에서
            // 입력을 받음)
            llvm::outs() << "[isIsolatedSubgraph] ExternalInput at ";
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
              llvm::outs() << "[isIsolatedSubgraph] ExternalOutput at ";
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

    while (status != IsolateStatus::Isolated) {
      if (detectionScope == 0) {
        llvm::outs() << "[match] scope exhausted\n";
        return subgraph{};
      }

      if (status == IsolateStatus::None) {
        backwardSearch();
        forwardSearch();
        status = isIsolatedSubgraph(expander, shrinker);
      } else if (status == IsolateStatus::ExternalInput) {
        llvm::outs() << "[match] status=" << toString(status)
                     << " -> expand backward\n";
        backwardSearch();
        status = isIsolatedSubgraph(expander, shrinker);
      } else if (status == IsolateStatus::ExternalOutput) {
        llvm::outs() << "[match] status=" << toString(status)
                     << " -> expand forward\n";
        forwardSearch();
        status = isIsolatedSubgraph(expander, shrinker);
      } else if (status == IsolateStatus::NotSplittable) {
        llvm::outs() << "[match] NotSplittable\n";
        return subgraph{};
      }
      llvm::outs() << "[match] status now=" << toString(status) << "\n";
    }

    llvm::outs() << "[match] SPLITTABLE: S=";
    printOpOneLine(expander);
    llvm::outs() << "  E=";
    printOpOneLine(shrinker);
    llvm::outs() << "\n";

    return subgraph{.s = expander,
        .e = shrinker,
        .subgraphNodes = getSubgraphNodes(expander, shrinker)};

    // S~E까지 몇 개의 op를 split해야하며
    // 최적화로 얻는 이득은 얼마인지?(peak memory reduction)
  }

  /*======================================================*/

  struct IsolatedSubgraph {
    Operation *entryGate = nullptr; // S (must be branching; fanout>=2)
    Operation *exitGate = nullptr;  // E
    llvm::SmallVector<Operation *, 256> nodes;
  };

  // -------------------- helpers --------------------
  static inline unsigned countDistinctUsers(Operation *op) {
    llvm::SmallPtrSet<Operation *, 16> uniq;
    for (Value r : op->getResults())
      for (Operation *u : r.getUsers())
        uniq.insert(u);
    return (unsigned)uniq.size();
  }
  static inline bool isBranching(Operation *op, unsigned minFanout = 2) {
    return countDistinctUsers(op) >= minFanout;
  }
  static Operation *nearestUpstreamBranching(
      Operation *start, unsigned minFanout = 2) {
    llvm::SmallPtrSet<Operation *, 32> vis;
    llvm::SmallVector<Operation *, 32> wl{start};
    while (!wl.empty()) {
      Operation *op = wl.pop_back_val();
      if (!vis.insert(op).second)
        continue;
      if (isBranching(op, minFanout))
        return op;
      for (Value in : op->getOperands())
        if (Operation *def = in.getDefiningOp())
          wl.push_back(def);
    }
    return nullptr;
  }
  static inline bool entryDominatesAll(Operation *entry, DominanceInfo &dom,
      const llvm::SmallPtrSetImpl<Operation *> &ops) {
    for (Operation *o : ops)
      if (!dom.dominates(entry, o))
        return false;
    return true;
  }
  static inline bool exitPostDominatesAll(Operation *exit,
      PostDominanceInfo &pdom, const llvm::SmallPtrSetImpl<Operation *> &ops) {
    for (Operation *o : ops)
      if (!pdom.postDominates(exit, o))
        return false;
    return true;
  }
  static Operation *promoteEntryIfNeeded(
      Operation *curEntry, Operation *witnessOp, DominanceInfo &dom) {
    if (dom.dominates(curEntry, witnessOp))
      return curEntry;
    return nearestUpstreamBranching(witnessOp); // nullptr이면 실패 신호
  }
  static Operation *demoteExitIfNeeded(
      Operation *curExit, Operation *witnessUser, PostDominanceInfo &pdom) {
    if (pdom.postDominates(curExit, witnessUser))
      return curExit;
    return witnessUser; // 간단 전략: 필요 시 더 뒤의 user로 내림
  }
  static void pruneToEssentialPath(Operation *entry, Operation *exit,
      llvm::SmallPtrSetImpl<Operation *> &scope) {
    llvm::SmallPtrSet<Operation *, 32> reachFromEntry, canReachExit, keep;
    // forward(users) from entry
    {
      llvm::SmallVector<Operation *, 128> wl{entry};
      while (!wl.empty()) {
        Operation *op = wl.pop_back_val();
        if (!scope.contains(op))
          continue;
        if (!reachFromEntry.insert(op).second)
          continue;
        for (Value r : op->getResults())
          for (Operation *u : r.getUsers())
            wl.push_back(u);
      }
    }
    // backward(defs) to exit
    {
      llvm::SmallVector<Operation *, 128> wl{exit};
      while (!wl.empty()) {
        Operation *op = wl.pop_back_val();
        if (!scope.contains(op))
          continue;
        if (!canReachExit.insert(op).second)
          continue;
        for (Value in : op->getOperands())
          if (Operation *def = in.getDefiningOp())
            wl.push_back(def);
      }
    }
    for (Operation *op : scope)
      if (reachFromEntry.contains(op) && canReachExit.contains(op))
        keep.insert(op);
    scope.clear();
    for (Operation *op : keep)
      scope.insert(op);
  }

  // -------------------- main (bool) --------------------
  static bool matchForkJoinPattern(Operation *peakOp,
      Operation *analysisRootOp, // 보통 func::FuncOp
      IsolatedSubgraph &out) {
    DominanceInfo dom(analysisRootOp);
    PostDominanceInfo pdom(analysisRootOp);

    // 1) 초기 S/E: S = seed에서 가장 가까운 분기(upstream), E = seed
    Operation *entryGate = nearestUpstreamBranching(peakOp, /*fanout=*/2);
    if (!entryGate)
      return false; // S must be branching
    Operation *exitGate = peakOp;

    // 2) seed 시작 lazy 확장 (게이트 위반 시 S/E를 ‘필요한 만큼’만 조정)
    llvm::SmallPtrSet<Operation *, 32> subgraphOps;
    llvm::SmallVector<Operation *, 32> worklist{peakOp};

    for (int iter = 0; iter < 256 && !worklist.empty(); ++iter) {
      Operation *op = worklist.pop_back_val();
      if (subgraphOps.contains(op))
        continue;

      // 게이트 조건 미충족 시 S/E 조정
      if (!dom.dominates(entryGate, op)) {
        if (Operation *ne = promoteEntryIfNeeded(entryGate, op, dom))
          entryGate = ne;
        else
          return false;
        if (!dom.dominates(entryGate, op))
          continue;
      }
      if (!pdom.postDominates(exitGate, op)) {
        if (Operation *nx = demoteExitIfNeeded(exitGate, op, pdom))
          exitGate = nx;
        else
          return false;
        if (!pdom.postDominates(exitGate, op))
          continue;
      }

      subgraphOps.insert(op);

      // 입력 제약: entry 제외 내부 op는 외부 producer 불가
      if (op != entryGate) {
        for (Value in : op->getOperands()) {
          if (Operation *def = in.getDefiningOp()) {
            if (!subgraphOps.contains(def)) {
              if (!dom.dominates(entryGate, def)) {
                if (Operation *ne = promoteEntryIfNeeded(entryGate, def, dom))
                  entryGate = ne;
                else
                  return false;
              }
              worklist.push_back(def);
            }
          } else {
            // BlockArgument → 외부 입력. 정책에 따라 강제 실패
            return false;
          }
        }
      }

      // 출력 제약: exit 제외 내부 op의 결과는 외부 user 불가
      if (op != exitGate) {
        for (Value r : op->getResults()) {
          for (Operation *u : r.getUsers()) {
            if (!subgraphOps.contains(u)) {
              if (!pdom.postDominates(exitGate, u)) {
                if (Operation *nx = demoteExitIfNeeded(exitGate, u, pdom))
                  exitGate = nx;
                else
                  return false;
              }
              worklist.push_back(u);
            }
          }
        }
      }
    }

    if (subgraphOps.empty() || !isBranching(entryGate, 2))
      return false;

    // 3) 더 좁게: U 내부에서 entry 뒤로, exit 앞으로 밀어 넣기
    {
      bool changed = true;
      for (int k = 0; k < 32 && changed; ++k) {
        changed = false;
        // entry tighten
        for (Operation *cand : subgraphOps) {
          if (cand == entryGate || !isBranching(cand))
            continue;
          if (entryDominatesAll(cand, dom, subgraphOps) &&
              dom.dominates(entryGate, cand)) {
            entryGate = cand;
            changed = true;
          }
        }
        // exit tighten
        for (Operation *cand : subgraphOps) {
          if (cand == exitGate)
            continue;
          if (exitPostDominatesAll(cand, pdom, subgraphOps) &&
              pdom.postDominates(cand, exitGate)) {
            exitGate = cand;
            changed = true;
          }
        }
      }
    }

    // 4) 경로 프루닝: Entry→…→Exit 경로에 실제로 기여하는 op만
    pruneToEssentialPath(entryGate, exitGate, subgraphOps);

    // 5) 최종 검증: 단일 entry/exit 격리
    for (Operation *op : subgraphOps)
      if (op != entryGate) {
        for (Value in : op->getOperands()) {
          if (Operation *def = in.getDefiningOp()) {
            if (!subgraphOps.contains(def))
              return false;
          } else
            return false;
        }
      }
    for (Operation *op : subgraphOps)
      if (op != exitGate) {
        for (Value r : op->getResults())
          for (Operation *u : r.getUsers())
            if (!subgraphOps.contains(u))
              return false;
      }

    // 6) 출력
    out.entryGate = entryGate;
    out.exitGate = exitGate;
    out.nodes.clear();
    out.nodes.insert(out.nodes.end(), subgraphOps.begin(), subgraphOps.end());
    return true;
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

    IsolateStatus status = isIsolatedSubgraph(concatOp, shrinker);

    while (status != IsolateStatus::Isolated) {
      if (status == IsolateStatus::ExternalInput) {
        if (!(concatOp = findConcatOpBackward())) {
          llvm::outs() << "[match Merge/Shrink] Cannot find Concat Op \n";
          return subgraph{};
        } else {
          llvm::outs() << "[match Merge/Shrink] found Concat Op: ";
          printOpOneLine(concatOp);
          llvm::outs() << '\n';
          status = isIsolatedSubgraph(concatOp, shrinker);
        }
      } else if (status == IsolateStatus::ExternalOutput) {
        if (!(shrinker = findShrinkerForward())) {
          llvm::outs() << "[match Merge/Shrink] Cannot find Shrinker \n";
          return subgraph{};
        } else {
          llvm::outs() << "[match Merge/Shrink] found Shrinker: ";
          printOpOneLine(shrinker);
          llvm::outs() << '\n';
          status = isIsolatedSubgraph(concatOp, shrinker);
        }
      } else if (status == IsolateStatus::NotSplittable) {
        llvm::outs() << "[match Merge/Shrink] status == NotSplittable \n";
        return subgraph{};
      }
    }

    return subgraph{.s = concatOp,
        .e = shrinker,
        .subgraphNodes = getSubgraphNodes(concatOp, shrinker)};
  }

  /*=========opimize logic=========*/
  static bool isDimPreserved(Operation *op, size_t dimension, Operation *e) {
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

      if (op == e) {
        return true;
      }

      Value outVal = convOp.getResult();
      for (Operation *userOp : outVal.getUsers()) {
        return isDimPreserved(userOp, dimension, e);
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
        }
        return false;
      }

      if (!isa<ONNXConstantOp>(opB)) {
        TensorType tensorTypeA = dyn_cast<TensorType>(a.getType());
        auto shape = tensorTypeA.getShape();
        // 뒤에서 두번째 차원
        if (shape.size() - 2 == dimension) {
          llvm::outs() << "[isDimPreserved] Failed: target dimension is "
                          "reduction dimension. Op: ";
          printOpOneLine(op);
          llvm::outs() << "\n";
        }
        return false;
      }

      if (op == e) {
        return true;
      }

      Value outVal = matmulOp.getResult();
      for (Operation *userOp : outVal.getUsers()) {
        return isDimPreserved(userOp, dimension, e);
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
              return isDimPreserved(userOp, i, e); // i == 바뀐 target dimension
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
    // elementwise 연산일 경우
    else if (isa<ONNXAddOp, ONNXClipOp>(op)) {
      if (op == e) {
        return true;
      }
      if (op->getNumResults() != 1) {
        llvm::outs() << "[isDimPreserved] Failed: Multiple results at Op: ";
        printOpOneLine(op);
        llvm::outs() << "\n";
        return false;
      }
      Value outVal = op->getResult(0);
      for (Operation *userOp : outVal.getUsers()) {
        return isDimPreserved(userOp, dimension, e);
      }
    }
    llvm::outs() << "[isDimPreserved] Failed: Unknown Op: ";
    printOpOneLine(op);
    llvm::outs() << "\n";
    return false;
  }

  llvm::DenseSet<int64_t> getSplittableDims(subgraph &subgraph) {
    llvm::DenseSet<int64_t> splittableDims;

    if (subgraph.s->getNumResults() != 1) {
      llvm::outs() << "[getSplittableDims] Fail: s has multiple results. \n";
    }

    Value sOut = subgraph.s->getResult(0);
    TensorType tensorType = dyn_cast<RankedTensorType>(sOut.getType());
    int64_t rank = tensorType.getRank();
    llvm::outs() << "[getSplittableDims] Rank: " << rank << "\n";

    auto userOps = sOut.getUsers();

    for (int i = 1; i < rank; i++) {
      bool preserved = false;

      for (Operation *op : userOps) {
        preserved = isDimPreserved(op, i, subgraph.e);
        if (!preserved) {
          break;
        }
      }
      if (preserved) {
        splittableDims.insert(i);
      }
    }

    return splittableDims;
  }

  int64_t determineSplitAxis(subgraph &subgraph) {
    if (subgraph.s->getNumResults() != 1) {
      llvm::outs() << "[determineSplitAxis] Fail: s has multiple results. \n";
    }

    Value sOutput = subgraph.s->getResult(0);
    auto tensorType = dyn_cast<TensorType>(sOutput.getType());
    ArrayRef<int64_t> shape = tensorType.getShape();

    // 1: Channel dimension (typically dim 1 for NCHW)
    int64_t dimension1 = shape[1];
    // s가 무슨 연산인지에 따라
    if (auto matmulOp = dyn_cast<ONNXMatMulOp>(subgraph.s)) {
      // TODO: MatMul 처리 로직

    } else if (auto convOp = dyn_cast<ONNXConvOp>(subgraph.s)) {
      // TODO: Conv 처리 로직

      // isa는 여러 타입을 동시에 확인 가능
    } else if (isa<ONNXAddOp, ONNXClipOp>(subgraph.s)) {
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

  void optimizeExpandShrinkPattern(subgraph expandShrinkSubgraph) {
    // TODO:
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
