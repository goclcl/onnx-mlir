#ifndef ONNX_MLIR_PEAKMEMORYANALYSIS_HPP
#define ONNX_MLIR_PEAKMEMORYANALYSIS_HPP

#include "mlir/Pass/Pass.h" // PassWrapper class, OperationPass template
#include <memory> // utilities for managing dynamic memory like std::unique_ptr

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createPeakMemoryAnalysis();
} // end of namespace onnx_mlir

#endif
