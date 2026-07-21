//===------ TensorSpillingPass.cpp - TinyMo tensor spilling ------===//
//
// TinyMo의 tensor spilling 단독 패스. 피크 시점에 라이브지만 접근되지 않는
// (long-living) 텐서 중 cold range가 가장 긴 것을 골라 spill(스토리지로) /
// fetch(되가져옴)를 삽입한다. 적용 전에 예상 새 피크를 계산해 개선이 없으면
// 적용하지 않고 종료한다. spill/fetch의 그래프 표현과 런타임은
// TinyMoUtils.hpp 참조.
//
//===----------------------------------------------------------------===//

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

struct TensorSpillingPass
    : public PassWrapper<TensorSpillingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorSpillingPass)

  StringRef getArgument() const override { return "onnx-tensor-spilling"; }

  StringRef getDescription() const override {
    return "Reduces peak memory by spilling long-living tensors to storage";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    int64_t spillId = 0;

    while (true) {
      Liveness liveness(funcOp);

      Operation *peakOp = nullptr;
      int64_t maxMemoryUsage = -1;
      funcOp.walk([&](Operation *op) {
        if (isa<ONNXConstantOp>(op) || isa<func::FuncOp>(op))
          return;
        int64_t usage =
            memoryUsageAtOp(op, liveness, /*includeConstants=*/false);
        if (usage > maxMemoryUsage) {
          maxMemoryUsage = usage;
          peakOp = op;
        }
      });
      if (!peakOp) {
        llvm::outs() << "[TensorSpillingPass] No peak op found.\n";
        break;
      }

      llvm::outs() << "[TensorSpillingPass] Peak memory usage (byte): "
                   << maxMemoryUsage << "\n";

      std::optional<SpillCandidate> cand =
          findSpillCandidate(funcOp, liveness, peakOp);
      if (!cand) {
        llvm::outs() << "[TensorSpillingPass] No spillable tensor at peak.\n";
        break;
      }

      // 예상 새 피크: victim이 라이브에서 빠지는 cold range 내부에서만
      // bytes만큼 줄어든다. 개선이 없으면 적용하지 않고 종료.
      int64_t expectedNewPeak =
          onnx_mlir::tinymo::expectedPeakAfterSpill(funcOp, liveness, *cand);
      if (expectedNewPeak >= maxMemoryUsage) {
        llvm::outs() << "[TensorSpillingPass] Spill would not reduce the "
                        "peak ("
                     << maxMemoryUsage << " -> " << expectedNewPeak
                     << " bytes). Stopping.\n";
        break;
      }

      llvm::outs() << "[TensorSpillingPass] Spilling tensor ("
                   << cand->bytes << " bytes, cold range " << cand->coldLen
                   << " ops) id=" << spillId << "\n";

      OpBuilder builder(funcOp.getContext());
      if (!applyTensorSpilling(*cand, builder, spillId))
        break;
      ++spillId;
    }
  }

};

} // anonymous namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createTensorSpillingPass() {
  return std::make_unique<TensorSpillingPass>();
}
} // namespace onnx_mlir
