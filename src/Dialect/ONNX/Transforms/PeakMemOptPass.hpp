#ifndef ONNX_MLIR_PEAKMEMOPTPASS_HPP
#define ONNX_MLIR_PEAKMEMOPTPASS_HPP

#include "mlir/Pass/Pass.h"
#include <memory>

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createPeakMemOptPass();
} 

#endif
