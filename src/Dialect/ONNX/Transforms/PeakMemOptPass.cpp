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
    Operation *s = nullptr;
    Operation *e = nullptr;
    const uint32_t DETECTION_SCOPE = 64;

    llvm::outs() << "[match] try EXPAND~SHRINK from peak, scope="
                 << DETECTION_SCOPE << "\n";

    bool matched = matchExpandShrinkPattern(peakOp, s, e, DETECTION_SCOPE);

    if (!matched) {
      llvm::outs() << "[match Expand/Shrink] no window\n";
    } else {
      llvm::outs() << "[match Expand/Shrink] SPLITTABLE window: S=";
      printOpOneLine(s);
      llvm::outs() << "  E=";
      printOpOneLine(e);
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
    if (matchMergeShrinkPattern(peakOp, s, e)) {
      llvm::outs() << "[match Merge/Shrink] SPLITTABLE window: S=";
      printOpOneLine(s);
      llvm::outs() << "  E=";
      printOpOneLine(e);
      llvm::outs() << "\n";
    } else {
      llvm::outs() << "[match Merge/Shrink] no window\n";
    }
    /* ---------------Merge/Shrink--------------- */
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
  static IsolateStatus isIsolatedSubGraph(Operation *S, Operation *E) {

    // S나 E를 못찾았으면 notSplittable
    if (!S || !E) {
      llvm::outs() << "[isIsolatedSubGraph] NotSplittable: S or E is nullptr\n";
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
    return IsolateStatus::Isolated;
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

    while (status != IsolateStatus::Isolated) {
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

  bool matchMergeShrinkPattern(
      Operation *peakOp, Operation *&sOut, Operation *&eOut) {

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
      return false;
    } else {
      llvm::outs() << "[match Merge/Shrink] found Concat Op: ";
      printOpOneLine(concatOp);
      llvm::outs() << '\n';
    }

    if (!(shrinker = findShrinkerForward())) {
      llvm::outs() << "[match Merge/Shrink] Cannot find Shrinker \n";
      return false;
    } else {
      llvm::outs() << "[match Merge/Shrink] found Shrinker: ";
      printOpOneLine(shrinker);
      llvm::outs() << '\n';
    }

    IsolateStatus status = isIsolatedSubGraph(concatOp, shrinker);

    while (status != IsolateStatus::Isolated) {
      if (status == IsolateStatus::ExternalInput) {
        if (!(concatOp = findConcatOpBackward())) {
          llvm::outs() << "[match Merge/Shrink] Cannot find Concat Op \n";
          return false;
        } else {
          llvm::outs() << "[match Merge/Shrink] found Concat Op: ";
          printOpOneLine(concatOp);
          llvm::outs() << '\n';
          status = isIsolatedSubGraph(concatOp, shrinker);
        }
      } else if (status == IsolateStatus::ExternalOutput) {
        if (!(shrinker = findShrinkerForward())) {
          llvm::outs() << "[match Merge/Shrink] Cannot find Shrinker \n";
          return false;
        } else {
          llvm::outs() << "[match Merge/Shrink] found Shrinker: ";
          printOpOneLine(shrinker);
          llvm::outs() << '\n';
          status = isIsolatedSubGraph(concatOp, shrinker);
        }
      } else if (status == IsolateStatus::NotSplittable) {
        llvm::outs() << "[match Merge/Shrink] status == NotSplittable \n";
        return false;
      }
    }

    sOut = concatOp;
    eOut = shrinker;
    return true;
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
