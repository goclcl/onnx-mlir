#include <fstream>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/TinyMoUtils.hpp"

using namespace mlir;
using namespace onnx_mlir::tinymo;

namespace {

// TinyMo의 tensor splitting 단독 패스. 분할 엔진은 TinyMoUtils로 추출되어
// 통합 드라이버(onnx-tinymo-opt)와 공유한다.
struct TensorSplittingPass
    : public PassWrapper<TensorSplittingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorSplittingPass)

  StringRef getArgument() const override { return "onnx-tensor-splitting"; }

  StringRef getDescription() const override {
    return "Reduces memory bottlenecks by splitting tensors";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    static bool hasLoggedOnce = false;

    // --- 로깅 로직 ---
    // 이 Pass가 처음 실행될 때 단 한 번만 실행됨.
    if (!hasLoggedOnce) {
      std::ofstream logFile(
          "memory_usage.txt", std::ios::out | std::ios::trunc);
      if (!logFile.is_open()) {
        llvm::errs() << "Error: Could not open memory_usage.txt for writing.\n";
        return;
      }

      llvm::outs()
          << "[TensorSplittingPass] Performing initial memory scan...\n";
      Liveness liveness(funcOp);
      funcOp.walk([&](Operation *op) {
        logFile << memoryUsageAtOp(op, liveness, /*includeConstants=*/true)
                << "\n";
      });

      logFile.close();
      llvm::outs() << "[TensorSplittingPass] Initial memory usage has been "
                      "written to memory_usage.txt\n";

      hasLoggedOnce = true;
    }

    // peak op 분석, 최적화를 반복함. 최적화 불가능하면 루프 종료
    while (1) {
      Liveness liveness(funcOp);

      Operation *peakOp = nullptr;
      int64_t maxMemoryUsage = -1;

      funcOp.walk([&](Operation *op) {
        int64_t currentOpMemoryUsage =
            memoryUsageAtOp(op, liveness, /*includeConstants=*/true);
        if (currentOpMemoryUsage > maxMemoryUsage) {
          maxMemoryUsage = currentOpMemoryUsage;
          peakOp = op;
        }
      });

      if (!peakOp) {
        llvm::outs()
            << "[TensorSplittingPass] No peak op found or function is empty.\n";
        break;
      }

      llvm::outs() << "[TensorSplittingPass] Peak memory Usage (byte): "
                   << maxMemoryUsage << "\n";

      std::optional<SplitCandidate> cand = findSplitCandidate(peakOp);
      if (!cand) {
        llvm::outs() << "[TensorSplittingPass] Peak op is not optimizable.\n";
        break;
      }

      OpBuilder builder(peakOp->getContext());
      if (applyTensorSplitting(*cand, builder)) {
        static int64_t optCount = 1;
        llvm::outs() << "[TensorSplittingPass] Optimization applied "
                     << optCount << " time" << (optCount == 1 ? "" : "s")
                     << ".\n\n";
        optCount++;
      } else {
        break;
      }
    } // End while
  }
}; // struct TensorSplittingPass

} // anonymous namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createTensorSplittingPass() {
  return std::make_unique<TensorSplittingPass>();
}
} // namespace onnx_mlir
