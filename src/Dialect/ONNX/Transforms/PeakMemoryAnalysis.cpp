#include <system_error>

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
#include "mlir/Support/LLVM.h" // isa/dyn_cast/cast

// ONNX ops
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {
/// PeakMemoryAnalysis Pass:
/// - 각 연산(Operation) 시점에서 살아있는 텐서들의 총 메모리 사용량을 계산.
/// - 가장 peak memory를 사용하는 Operation을 찾아 출력 및 파일로 저장.
struct PeakMemoryAnalysis
    : public PassWrapper<PeakMemoryAnalysis, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PeakMemoryAnalysis)

  StringRef getArgument() const override { return "onnx-peak-memory-analysis"; }
  StringRef getDescription() const override {
    return "Analyzes peak memory usage.";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    Liveness &liveness = getAnalysis<Liveness>();

    Operation *peakOp = nullptr;
    int64_t maxMemoryUsage = -1;

    // 로그 파일 (상수 포함)
    std::error_code ec1;
    llvm::raw_fd_ostream logFileWithConstants("memory_report.txt", ec1);
    if (ec1) {
      llvm::errs() << "Error opening log file (with constants): " << ec1.message() << "\n";
      return;
    }

    // 로그 파일 (상수 제외)
    std::error_code ec2;
    llvm::raw_fd_ostream logFileWithoutConstants("memory_report_no_constants.txt", ec2);
    if (ec2) {
      llvm::errs() << "Error opening log file (without constants): " << ec2.message() << "\n";
      return;
    }

    logFileWithConstants << "Operation-wise Memory Usage Report (with Constants)\n";
    logFileWithConstants << "====================================================\n";
    logFileWithoutConstants << "Operation-wise Memory Usage Report (without Constants)\n";
    logFileWithoutConstants << "=======================================================\n";

    // 모든 연산에 대해 메모리 사용량 계산 및 기록 (onnx.Constant 제외)
    funcOp.walk([&](Operation *op) {
      // onnx.Constant와 func.func op는 스킵함
      if (mlir::isa<mlir::ONNXConstantOp>(op) || mlir::isa<mlir::func::FuncOp>(op))
        return;

      // 1. 상수 포함해서 메모리 계산&로깅
      int64_t usageWithConstants = calculateMemoryUsageAtOp(op, liveness, /*excludeConstants=*/false);
      logFileWithConstants << "Memory: " << usageWithConstants << " bytes | Op: ";
      op->print(logFileWithConstants);
      logFileWithConstants << "\n---------------------------------\n";

      // Peak 메모리는 상수를 포함한 전체 사용량 기준으로 계산
      if (usageWithConstants > maxMemoryUsage) {
        maxMemoryUsage = usageWithConstants;
        peakOp = op;
      }

      // 2. 상수 제외 메모리 계산 및 로깅
      int64_t usageWithoutConstants = calculateMemoryUsageAtOp(op, liveness, /*excludeConstants=*/true);
      logFileWithoutConstants << "Memory: " << usageWithoutConstants << " bytes | Op: ";
      op->print(logFileWithoutConstants);
      logFileWithoutConstants << "\n---------------------------------\n";
      // =============================================================
    });

    // Peak memory 결과를 콘솔과 로그 파일에 출력
    if (peakOp) {
      // 1. 콘솔과 상수 포함 로그 파일에 결과 출력
      llvm::outs() << "[PeakMemoryAnalysis] Peak memory usage: "
                   << maxMemoryUsage << " bytes\n";
      llvm::outs() << "[PeakMemoryAnalysis] At operation: " << *peakOp << "\n";

      logFileWithConstants << "\n\nPeak Memory Usage: " << maxMemoryUsage << " bytes\n";
      logFileWithConstants << "At Operation:\n";
      peakOp->print(logFileWithConstants);
      logFileWithConstants << "\n";
      // peak 시점에 살아있는 value들 출력
      logFileWithConstants << "Live value lists:" << "\n";
      const LivenessBlockInfo *blockLiveness =
        liveness.getLiveness(peakOp->getBlock());
      for (Value val : blockLiveness->currentlyLiveValues(peakOp)) {
        Operation *defOp = val.getDefiningOp();
        if (defOp && mlir::isa<ONNXConstantOp>(defOp))
        continue;
        logFileWithConstants << val << "\n";
      }


      // 2. 상수 제외 로그 파일에 상수 포함한 분석의 peak op 정보 추가
      // peakOp 시점의 메모리 사용량을 상수 제외 기준으로 다시 계산
      int64_t peakOpUsageWithoutConstants = calculateMemoryUsageAtOp(peakOp, liveness, /*excludeConstants=*/true);

      // peak op자체는 상수를 포함 했을때의 peak op이고,
      // usage는 해당 op의 상수를 제외 했을때의 계산 결과임
      logFileWithoutConstants << "\n\nPeak Memory Usage: " << peakOpUsageWithoutConstants << " bytes\n";
      logFileWithoutConstants << "At Operation:\n";
      peakOp->print(logFileWithoutConstants);
      logFileWithoutConstants << "\n";

    } else {
      llvm::outs() << "[PeakMemoryAnalysis] No operation found.\n";
    }
  }

private:
  /// 주어진 TensorType의 메모리 크기를 byte 단위로 계산
  /// - 동적 차원은 defaultDynamicSize로 추정
  int64_t getTensorSize(TensorType tensorType, int64_t defaultDynamicSize = 1) {
    // 유효하지 않은 타입이거나 랭크가 없는 경우 0 반환
    if (!tensorType || !tensorType.hasRank())
      return 0;

    // 요소의 bit width 추출
    Type elementType = tensorType.getElementType();
    unsigned bitWidth = 0;

    if (elementType.isIntOrIndex()) {
      // 정수 또는 index 타입인 경우
      bitWidth = elementType.getIntOrFloatBitWidth();
    } else if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
      // float 타입인 경우
      bitWidth = floatType.getWidth();
    } else {
      // 지원하지 않는 데이터 타입
      return 0;
    }

    // 전체 요소 개수 계산
    int64_t totalElements = 1;
    for (int64_t dim : tensorType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        // 동적 차원은 기본값으로 대체하여 추정
        totalElements *= defaultDynamicSize;
      } else if (dim == 0) {
        // 크기가 0인 차원이 있는 경우 전체 크기는 0
        return 0;
      } else {
        // 정적 차원은 그대로 곱함
        totalElements *= dim;
      }
    }

    // 총 요소 개수 × 요소 크기 (bit → byte)
    return (totalElements * bitWidth) / 8;
  }

  /// 지정된 Operation에서 살아있는 텐서들의 총 메모리 사용량 계산
  int64_t calculateMemoryUsageAtOp(Operation *op, Liveness &liveness, bool excludeConstants) {
    const LivenessBlockInfo *blockLiveness =
        liveness.getLiveness(op->getBlock());
    if (!blockLiveness)
      return 0;

    int64_t memoryUsage = 0;
    SmallPtrSet<Value, 16> liveValues;

    // 현재 시점에 live한 값 + 현재 op의 결과 값을 모두 고려
    for (Value val : blockLiveness->currentlyLiveValues(op))
      liveValues.insert(val);
    for (Value result : op->getResults())
      liveValues.insert(result);

    // 텐서 타입인 경우에만 메모리 크기 합산
    for (Value val : liveValues) {
      // 상수 제외 조건문
      Operation *defOp = val.getDefiningOp();
      if (excludeConstants && defOp && mlir::isa<ONNXConstantOp>(defOp))
        continue;

      if (auto tensorType = mlir::dyn_cast<TensorType>(val.getType()))
        memoryUsage += getTensorSize(tensorType);
    }

    return memoryUsage;
  }
};

} // namespace

// Pass 생성 함수 등록
namespace onnx_mlir {
std::unique_ptr<Pass> createPeakMemoryAnalysis() {
  return std::make_unique<PeakMemoryAnalysis>();
}
} // namespace onnx_mlir
