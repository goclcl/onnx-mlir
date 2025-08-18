#include <system_error>
#include <vector>
#include <algorithm>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {
struct PeakMemoryAnalysis
    : public PassWrapper<PeakMemoryAnalysis, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PeakMemoryAnalysis)

  StringRef getArgument() const override { return "onnx-peak-memory-analysis"; }
  StringRef getDescription() const override {
    return "Analyzes peak memory usage (Parameters as baseline, Activations as live).";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    Liveness &liveness = getAnalysis<Liveness>();

    // 상수 baseline 계산
    const int64_t baselineBytes = computeConstantBaseline(funcOp);

    // 로그 파일
    std::error_code ec;
    llvm::raw_fd_ostream logFile("memory_report.txt", ec);
    if (ec) {
      llvm::errs() << "Error opening log file: " << ec.message() << "\n";
      signalPassFailure();
      return;
    }

    logFile << "Operation-wise Memory Usage Report\n";
    logFile << "=================================\n";
    logFile << "Parameters (baseline): " << baselineBytes << " bytes\n\n";

    // Peak 탐색
    Operation *peakOp = nullptr;
    int64_t peakTotal = -1;
    int64_t peakActivations = -1;

    funcOp.walk([&](Operation *op) {
      // onnx.Constant와 func.func op는 스킵
      if (isa<ONNXConstantOp>(op) || isa<func::FuncOp>(op))
        return;

      int64_t liveActivBytes = calculateActivationUsageAtOp(op, liveness);
      int64_t totalBytes = baselineBytes + liveActivBytes;

      logFile << "Total: " << totalBytes
              << " bytes | Activations: " << liveActivBytes
              << " bytes | Op: ";
      op->print(logFile);
      logFile << "\n---------------------------------\n";

      if (totalBytes > peakTotal) {
        peakTotal = totalBytes;
        peakActivations = liveActivBytes;
        peakOp = op;
      }
    });

    if (!peakOp) {
      llvm::outs() << "[PeakMemoryAnalysis] No operation found.\n";
      return;
    }

    // 요약 출력
    llvm::outs() << "[PeakMemoryAnalysis] Peak TOTAL memory: "
                 << peakTotal << " bytes (Parameters: " << baselineBytes
                 << " + Activations: " << peakActivations << ")\n";
    llvm::outs() << "[PeakMemoryAnalysis] At operation: " << *peakOp << "\n";

    logFile << "\n\n=== PEAK SUMMARY ===\n";
    logFile << "Peak TOTAL: " << peakTotal << " bytes\n";
    logFile << "  - Parameters (baseline): " << baselineBytes << " bytes\n";
    logFile << "  - Activations (live):    " << peakActivations << " bytes\n";
    logFile << "At Operation:\n";
    peakOp->print(logFile);
    logFile << "\n";

    // 피크 시점 live activation value 정렬 출력 (크기 내림차순)
    const LivenessBlockInfo *blk = liveness.getLiveness(peakOp->getBlock());
    if (!blk) {
      llvm::errs() << "[PeakMemoryAnalysis] Missing liveness for peak block.\n";
      signalPassFailure();
      return;
    }

    std::vector<std::pair<int64_t, Value>> liveList;
    SmallPtrSet<Value, 16> liveVals;
    for (Value v : blk->currentlyLiveValues(peakOp)) liveVals.insert(v);

    for (Value v : liveVals) {
      if (Operation *def = v.getDefiningOp()) {
        if (isa<ONNXConstantOp>(def)) continue; // 상수 제외
      }
      auto tt = dyn_cast<TensorType>(v.getType());
      if (!tt) continue;

      int64_t size = getTensorSize(tt); // dynamic dim이면 에러
      if (size == -1) {
        signalPassFailure();
        return;
      }
      liveList.emplace_back(size, v);
    }

    std::sort(liveList.begin(), liveList.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    logFile << "Live activation values at peak (sorted by bytes desc):\n";
    for (auto &kv : liveList) {
      int64_t bytes = kv.first;
      Value v = kv.second;
      logFile << "  [" << bytes << " bytes]  " << v << " : " << v.getType() << "\n";
    }
  }

private:
  // ===== 유틸 =====

  // TensorType 크기 계산: 동적 차원이 있으면 에러 반환
  static int64_t getTensorSize(TensorType tensorType) {
    if (!tensorType || !tensorType.hasRank())
      return -1;

    Type elem = tensorType.getElementType();
    unsigned bitWidth = 0;
    if (elem.isIntOrIndex()) {
      bitWidth = elem.getIntOrFloatBitWidth();
    } else if (auto ft = dyn_cast<FloatType>(elem)) {
      bitWidth = ft.getWidth();
    } else {
      return -1;
    }

    int64_t total = 1;
    for (int64_t d : tensorType.getShape()) {
      if (d == ShapedType::kDynamic) {
        // 동적 차원 발견 시 에러
        llvm::errs() << "[PeakMemoryAnalysis] Error: dynamic dimension found in type: "
                     << tensorType << "\n";
        return -1;
      }
      total *= d;
    }
    return (total * bitWidth) / 8;
  }

  // 함수 내 ONNXConstantOp 결과 텐서 크기의 총합(baseline).
  static int64_t computeConstantBaseline(func::FuncOp funcOp) {
    int64_t sum = 0;
    SmallPtrSet<Operation *, 16> seen; // 동일 op 중복 방지
    funcOp.walk([&](ONNXConstantOp cst) {
      Operation *op = cst.getOperation();
      if (!seen.insert(op).second) return;
      for (Value r : op->getResults()) {
        if (auto tt = dyn_cast<TensorType>(r.getType()))
          sum += getTensorSize(tt);
      }
    });
    return sum;
  }

  // 지정 Op 시점 라이브 activation 바이트 합 (상수 제외)
   static int64_t calculateActivationUsageAtOp(Operation *op, Liveness &liveness) {
    const LivenessBlockInfo *blk = liveness.getLiveness(op->getBlock());
    if (!blk) return 0;

    int64_t bytes = 0;
    SmallPtrSet<Value, 16> live;

    // live values
    for (Value v : blk->currentlyLiveValues(op)) live.insert(v);
    // 현재 op의 결과 values
    for (Value r : op->getResults()) live.insert(r);

    for (Value v : live) {
      if (Operation *def = v.getDefiningOp()) {
        if (isa<ONNXConstantOp>(def)) continue; // 상수 제외
      }
      if (auto tt = dyn_cast<TensorType>(v.getType()))
        bytes += getTensorSize(tt);
    }
    return bytes;
  }
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createPeakMemoryAnalysis() {
  return std::make_unique<PeakMemoryAnalysis>();
}
} // namespace onnx_mlir

