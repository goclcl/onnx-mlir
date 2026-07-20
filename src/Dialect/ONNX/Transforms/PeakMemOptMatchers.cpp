//===------ PeakMemOptMatchers.cpp - 패턴 matcher 구현 ------===//

#include "src/Dialect/ONNX/Transforms/PeakMemOptMatchers.hpp"

#include <queue>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace peakmem {

namespace {

// expand/shrink와 fork/join이 공유하는 확장 루프: 후보 S/E로 만든 서브그래프가
// 고립(Isolated)될 때까지, 고립을 깨는 방향(ExternalInput→backward,
// ExternalOutput→forward)으로 탐색을 넓힌다.
//  - backward/forward: 각 matcher의 후보 탐색(내부 상태로 S/E 후보를 갱신)
//  - makeCandidate: 현재 후보로 서브그래프 구성
//  - exhausted: true면 탐색 여지 소진 → 빈 서브그래프 반환
// 후보를 못 찾아 빈 서브그래프가 되면(고립 검사는 빈 집합을 Isolated로 봄)
// 그대로 빈 서브그래프가 반환되므로, 호출자는 getS()로 성공 여부를 가린다.
Subgraph growUntilIsolated(StringRef tag,
    llvm::function_ref<void()> backwardSearch,
    llvm::function_ref<void()> forwardSearch,
    llvm::function_ref<Subgraph()> makeCandidate,
    llvm::function_ref<bool()> exhausted) {
  IsolateStatus status = IsolateStatus::None;
  while (status != IsolateStatus::Isolated) {
    if (exhausted()) {
      pmoDbg() << "[" << tag << "] search exhausted\n";
      return Subgraph{};
    }

    if (status == IsolateStatus::None) {
      backwardSearch();
      forwardSearch();
      status = checkIsolation(makeCandidate());
    } else if (status == IsolateStatus::ExternalInput) {
      pmoDbg() << "[" << tag << "] status=" << toString(status)
               << " -> expand backward\n";
      backwardSearch();
      status = checkIsolation(makeCandidate());
    } else if (status == IsolateStatus::ExternalOutput) {
      pmoDbg() << "[" << tag << "] status=" << toString(status)
               << " -> expand forward\n";
      forwardSearch();
      status = checkIsolation(makeCandidate());
    } else { // NotSplittable
      pmoDbg() << "[" << tag << "] NotSplittable\n";
      return Subgraph{};
    }
    pmoDbg() << "[" << tag << "] status now=" << toString(status) << "\n";
  }
  return makeCandidate();
}

bool isForkOp(Operation *op) {
  if (op->getNumResults() != 1)
    return false;
  return op->getResult(0).hasNUsesOrMore(2);
}

bool isJoinOp(Operation *op, const llvm::SmallPtrSet<Operation *, 4> &branches) {
  for (Operation *branch : branches) {
    llvm::DenseSet<Operation *> nodes = getForwardReachableNodes(branch, op);
    if (!nodes.contains(op))
      return false;
  }
  return true;
}

} // namespace

Subgraph matchExpandShrinkPattern(Operation *peakOp) {
  Operation *expander = nullptr;
  Operation *shrinker = nullptr;
  llvm::SmallPtrSet<Operation *, 32> visited;
  std::queue<Operation *> backwardWorklist;
  std::queue<Operation *> forwardWorklist;
  bool foundExpander = false;
  bool foundShrinker = false;

  pmoDbg() << "[matchExpandShrinkPattern] start: peak=";
  printOpOneLine(peakOp);
  pmoDbg() << "\n";

  // peakOp를 worklist에 추가
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

    while (!backwardWorklist.empty()) {
      Operation *currentOp = backwardWorklist.front();
      if (!currentOp)
        continue;
      backwardWorklist.pop();

      // 오프라인으로 접히는 생산자(상수/NoValue/상수의 Split)라면 skip
      if (isOfflineProducer(currentOp))
        continue;

      for (Value operand : currentOp->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          if (visited.insert(defOp).second)
            backwardWorklist.push(defOp); // 상위 op를 worklist에 추가
        }
      }

      // 체인 헤드 조건: 출력이 곧장 갈라지는(사용자 ≥2) op는 단일 체인의
      // 시작이 될 수 없다 — 그런 fork 생산자는 fork/join 패턴의 소관이므로
      // S 후보로 잡지 않고 탐색을 더 위로 계속한다.
      if (checkTensorSizeChange(currentOp) == TensorSizeChange::EXPAND &&
          currentOp->getNumResults() == 1 &&
          currentOp->getResult(0).hasOneUse()) {
        expander = currentOp;
        foundExpander = true;
        pmoDbg() << "    [backwardSearch] EXPAND candidate: ";
        printOpOneLine(expander);
        pmoDbg() << "\n";
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

    while (!forwardWorklist.empty()) {
      Operation *currentOp = forwardWorklist.front();
      forwardWorklist.pop();

      if (isa<ONNXConstantOp>(currentOp))
        continue;

      pmoDbg() << "    [forwardSearch] pop ";
      printOpOneLine(currentOp);
      pmoDbg() << "\n";

      for (Value result : currentOp->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (visited.insert(user).second)
            forwardWorklist.push(user);
        }
      }

      if (checkTensorSizeChange(currentOp) == TensorSizeChange::SHRINK) {
        shrinker = currentOp;
        foundShrinker = true;
        pmoDbg() << "    [forwardSearch] SHRINK candidate: ";
        printOpOneLine(shrinker);
        pmoDbg() << "\n";
        break;
      }
    }
  };

  Subgraph result = growUntilIsolated("matchExpandShrinkPattern",
      backwardSearch, forwardSearch,
      [&]() { return Subgraph(expander, shrinker); },
      []() { return false; });
  if (!result.getS())
    return Subgraph{};

  // expand/shrink는 "단일 체인" 패턴이다: 영역 안에 fork가 있으면 이 영역은
  // 체인이 아니므로 매칭이 아니다. (S를 아무리 위로 올려도 fork는 영역 안에
  // 남으므로 이 시점에서 종료가 맞다.)
  for (Operation *op : result.subgraphNodes) {
    if (op == result.getE())
      continue;
    for (Value r : op->getResults()) {
      if (!r.hasOneUse()) {
        pmoDbg() << "[matchExpandShrinkPattern] region is not a single "
                    "chain -> no match\n";
        return Subgraph{};
      }
    }
  }

  pmoDbg() << "[matchExpandShrinkPattern] SPLITTABLE: S=";
  printOpOneLine(result.getS());
  pmoDbg() << "  E=";
  printOpOneLine(result.getE());
  pmoDbg() << "\n";

  return result;
}

Subgraph matchForkJoinPattern(Operation *peakOp) {
  Operation *forkOp = nullptr;
  Operation *joinOp = nullptr;
  bool foundFork = false;
  bool foundJoin = false;
  llvm::SmallPtrSet<Operation *, 32> visited;
  std::queue<Operation *> backwardWorklist;
  std::queue<Operation *> forwardWorklist;
  llvm::SmallPtrSet<Operation *, 4> branches;

  auto backwardSearch = [&]() {
    // 기존 후보 S가 존재했다면 초기화
    if (foundFork) {
      forkOp = nullptr;
      foundFork = false;
      branches.clear();
    }

    while (!backwardWorklist.empty()) {
      Operation *currentOp = backwardWorklist.front();
      if (!currentOp)
        continue;
      backwardWorklist.pop();

      // 오프라인으로 접히는 생산자(상수/NoValue/상수의 Split)라면 skip
      if (isOfflineProducer(currentOp)) {
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
        pmoDbg() << "    [backwardSearch] forkOp candidate: ";
        printOpOneLine(forkOp);
        pmoDbg() << "\n";
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

      pmoDbg() << "    [forwardSearch] pop ";
      printOpOneLine(currentOp);
      pmoDbg() << "\n";

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
        pmoDbg() << "    [forwardSearch] joinOp candidate: ";
        printOpOneLine(joinOp);
        pmoDbg() << "\n";
        break;
      }
    }
  };

  pmoDbg() << "[matchForkJoinPattern] start: peak=";
  printOpOneLine(peakOp);
  pmoDbg() << "\n";

  // peakOp를 worklist에 추가
  if (visited.insert(peakOp).second) {
    backwardWorklist.push(peakOp);
    forwardWorklist.push(peakOp);
  }

  Subgraph result = growUntilIsolated("matchForkJoinPattern", backwardSearch,
      forwardSearch, [&]() { return Subgraph(forkOp, joinOp); },
      [&]() { return false; });
  if (!result.getS()) {
    pmoDbg() << "[matchForkJoinPattern] no fork/join found\n";
    return Subgraph{};
  }

  pmoDbg() << "[matchForkJoinPattern] SPLITTABLE: S=";
  printOpOneLine(result.getS());
  pmoDbg() << "  E=";
  printOpOneLine(result.getE());
  pmoDbg() << "\n";

  return result;
}

Subgraph matchMergeShrinkPattern(Operation *peakOp) {

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
        // defOp가 null이거나 오프라인으로 접히는 생산자면 스킵
        // 이미 방문한 노드도 스킵
        if (!defOp || findConcatVisited.contains(defOp) ||
            isOfflineProducer(defOp))
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
    pmoDbg() << "[match Concat-Shrink] Cannot find Concat Op \n";
    return Subgraph{};
  } else {
    pmoDbg() << "[match Concat-Shrink] found Concat Op: ";
    printOpOneLine(concatOp);
    pmoDbg() << '\n';
  }

  if (!(shrinker = findShrinkerForward())) {
    pmoDbg() << "[match Concat-Shrink] Cannot find Shrinker \n";
    return Subgraph{};
  } else {
    pmoDbg() << "[match Concat-Shrink] found Shrinker: ";
    printOpOneLine(shrinker);
    pmoDbg() << '\n';
  }

  // 이 matcher의 확장은 "탐색 넓히기"가 아니라 "다음 후보 찾기"라
  // growUntilIsolated 대신 고유의 재시도 루프를 쓴다.
  IsolateStatus status = checkIsolation(Subgraph(concatOp, shrinker));

  while (status != IsolateStatus::Isolated) {
    if (status == IsolateStatus::ExternalInput) {
      if (!(concatOp = findConcatOpBackward())) {
        pmoDbg() << "[match Concat-Shrink] Cannot find Concat Op \n";
        return Subgraph{};
      } else {
        pmoDbg() << "[match Concat-Shrink] found Concat Op: ";
        printOpOneLine(concatOp);
        pmoDbg() << '\n';
        status = checkIsolation(Subgraph(concatOp, shrinker));
      }
    } else if (status == IsolateStatus::ExternalOutput) {
      if (!(shrinker = findShrinkerForward())) {
        pmoDbg() << "[match Concat-Shrink] Cannot find Shrinker \n";
        return Subgraph{};
      } else {
        pmoDbg() << "[match Concat-Shrink] found Shrinker: ";
        printOpOneLine(shrinker);
        pmoDbg() << '\n';
        status = checkIsolation(Subgraph(concatOp, shrinker));
      }
    } else if (status == IsolateStatus::NotSplittable) {
      pmoDbg() << "[match Concat-Shrink] status == NotSplittable \n";
      return Subgraph{};
    }
  }

  return Subgraph(concatOp, shrinker);
}

} // namespace peakmem
} // namespace onnx_mlir
