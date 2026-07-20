# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_rewriter(Decompose)
add_onnx_mlir_rewriter(ConstProp)
add_onnx_mlir_rewriter(ConvOpt)

add_onnx_mlir_library(OMShapeInference
  ShapeInference.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMShapeInferenceOpInterface
  MLIRFuncDialect
  )

add_onnx_mlir_library(OMShapeInferencePass
  ShapeInferencePass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMShapeInferenceOpInterface
  MLIRFuncDialect
  MLIRPass
  OMShapeInference
  )

add_onnx_mlir_library(OMInstrumentONNX
  InstrumentONNXSignaturePass.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMONNXOps
  OMKrnlOps
  MLIRPass
  OMOptionUtils
  )

add_onnx_mlir_library(OMONNXRewrite
  ConstProp.cpp
  ConvOpt.cpp
  Decompose.cpp
  DecomposeEinsum.cpp
  ScrubDisposablePass.cpp
  SetONNXNodeName.cpp
  Recompose.cpp

  DEPENDS
  OMONNXDecomposeIncGen
  OMONNXConstPropIncGen
  OMONNXConvOptIncGen

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  OMONNXOps
  )

add_onnx_mlir_library(OMOpTransform
  ONNXOpTransformPass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRPass
  OMONNXRewrite
  OMShapeInferencePass
  MLIRTransforms
  )

add_onnx_mlir_library(OMHybridTransform
  ONNXHybridTransformPass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMShapeInferenceOpInterface
  OMONNXRewrite
  MLIRPass
  MLIRTransforms
  OMShapeInference
  )

add_onnx_mlir_library(OMONNXPreKrnlVerifyONNX
  ONNXPreKrnlVerifyPass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRPass
  OMSupport
  )

add_onnx_mlir_library(OMONNXSimplifyShapeRelatedOps
  SimplifyShapeRelatedOps.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMONNXRewrite
  OMShapeInferencePass
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  )

add_onnx_mlir_library(OMONNXStandardFuncReturnPass
  StandardFuncReturnPass.cpp

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  MLIRFuncDialect
  OMONNXOps
  OMShapeInference
  )

add_onnx_mlir_library(OMPeakMemOptPass
  PeakMemOptPass.cpp
  PeakMemOptUtils.cpp
  PeakMemOptMatchers.cpp
  PeakMemOptPlanner.cpp
  PeakMemOptEngine.cpp
  PeakMemOptRules.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRAnalysis
    MLIRFuncDialect
    MLIRTransformUtils
    MLIRTransforms
    OMONNXOps
    OMShapeInference
    OMSupport
)

add_onnx_mlir_library(OMPeakMemoryAnalysis
  PeakMemoryAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect          # func::FuncOp 사용
  MLIRAnalysis             # Liveness 등 Analysis
  MLIRIR                   # IR Core (BuiltinTypes 등)
  MLIRPass                 # Pass 등록
  MLIRSupport              # LLVM ADT 포함
  OMONNXOps                # ONNX 연산
)

