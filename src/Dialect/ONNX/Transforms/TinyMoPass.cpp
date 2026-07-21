//===------ TinyMoPass.cpp - TinyMo 통합 드라이버 ------===//
//
// TinyMo(baseline) 재현: 매 iteration 피크를 찾고, tensor splitting과
// tensor spilling 두 방법의 기대 감소량을 비교해 큰 쪽을 적용한다
// (논문 III-B: "applies one of the optimization methods considering their
// expected memory reductions"). 어느 쪽도 불가능하거나 피크 개선이 없으면
// 종료한다. 메모리 계정은 TinyMo를 따라 activation만 센다(파라미터는
// 플래시 상주).
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

struct TinyMoPass
    : public PassWrapper<TinyMoPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TinyMoPass)

  StringRef getArgument() const override { return "onnx-tinymo-opt"; }

  StringRef getDescription() const override {
    return "TinyMo memory optimizer: tensor spilling + tensor splitting, "
           "selected per iteration by expected reduction";
  }

  Option<int> maxItersOpt{*this, "max-iters",
      llvm::cl::desc("Stop after this many applied optimizations "
                     "(default: repeat until no improvement)."),
      llvm::cl::init(128)};

  TinyMoPass() = default;
  TinyMoPass(const TinyMoPass &pass)
      : PassWrapper<TinyMoPass, OperationPass<func::FuncOp>>() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    int64_t spillId = 0;
    int64_t prevPeak = -1;
    // 상한: max-iters 옵션 (기본 128 = 사실상 후보 소진까지)
    const int maxIters = maxItersOpt;

    for (int iter = 0; iter < maxIters; ++iter) {
      Liveness liveness(funcOp);

      Operation *peakOp = nullptr;
      int64_t peakBytes = -1;
      funcOp.walk([&](Operation *op) {
        if (isa<ONNXConstantOp>(op) || isa<func::FuncOp>(op))
          return;
        int64_t usage =
            memoryUsageAtOp(op, liveness, /*includeConstants=*/false);
        if (usage > peakBytes) {
          peakBytes = usage;
          peakOp = op;
        }
      });
      if (!peakOp)
        break;

      llvm::outs() << "[TinyMo] iteration " << iter << ": peak " << peakBytes
                   << " bytes\n";
      if (prevPeak >= 0 && peakBytes >= prevPeak) {
        llvm::outs() << "[TinyMo] peak did not improve; stopping.\n";
        break;
      }

      // 두 방법의 후보와 "예상 전역 피크 개선량". 같은 피크값 지점이
      // 여럿이면 국소 감소가 전역 피크를 못 낮출 수 있으므로, 전역
      // 시뮬레이션으로 게이트한다 — 개선 없는 변환은 적용하지 않는다.
      std::optional<SplitCandidate> splitCand = findSplitCandidate(peakOp);
      std::optional<SpillCandidate> spillCand =
          findSpillCandidate(funcOp, liveness, peakOp);
      int64_t splitRed = -1, spillRed = -1;
      if (splitCand) {
        splitRed =
            peakBytes - expectedPeakAfterSplit(funcOp, liveness, *splitCand);
        if (splitRed <= 0) {
          llvm::outs() << "[TinyMo] splitting would not reduce the global "
                          "peak; dropped\n";
          splitCand.reset();
          splitRed = -1;
        }
      }
      if (spillCand) {
        spillRed =
            peakBytes - expectedPeakAfterSpill(funcOp, liveness, *spillCand);
        if (spillRed <= 0) {
          llvm::outs() << "[TinyMo] spilling would not reduce the global "
                          "peak; dropped\n";
          spillCand.reset();
          spillRed = -1;
        }
      }

      if (!splitCand && !spillCand) {
        llvm::outs() << "[TinyMo] no applicable optimization at peak; "
                        "stopping.\n";
        break;
      }

      OpBuilder builder(funcOp.getContext());
      bool applied = false;
      if (splitRed >= spillRed) {
        llvm::outs() << "[TinyMo] apply tensor splitting (expected -"
                     << splitRed << " bytes"
                     << (spillCand ? (" vs spilling -" +
                                         std::to_string(spillRed) + " bytes")
                                   : std::string(""))
                     << ")\n";
        applied = applyTensorSplitting(*splitCand, builder);
      } else {
        llvm::outs() << "[TinyMo] apply tensor spilling (expected -"
                     << spillRed << " bytes"
                     << (splitCand ? (" vs splitting -" +
                                         std::to_string(splitRed) + " bytes")
                                   : std::string(""))
                     << ", cold range " << spillCand->coldLen << " ops)\n";
        applied = applyTensorSpilling(*spillCand, builder, spillId++);
      }
      if (!applied) {
        llvm::outs() << "[TinyMo] transform failed; stopping.\n";
        break;
      }
      prevPeak = peakBytes;
    }

    // 최종 피크 보고
    Liveness liveness(funcOp);
    int64_t finalPeak = 0;
    funcOp.walk([&](Operation *op) {
      if (isa<ONNXConstantOp>(op) || isa<func::FuncOp>(op))
        return;
      finalPeak = std::max(
          finalPeak, memoryUsageAtOp(op, liveness, /*includeConstants=*/false));
    });
    llvm::outs() << "[TinyMo] done: final peak " << finalPeak << " bytes\n";
  }
};

} // anonymous namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createTinyMoPass() {
  return std::make_unique<TinyMoPass>();
}
} // namespace onnx_mlir
