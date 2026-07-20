// RUN: onnx-mlir-opt --onnx-peak-mem-opt --shape-inference %s | FileCheck %s

// Same graph as onnx_peak_mem_opt_expand_shrink_conv.mlir but with the
// depthwise weights (%w2/%b2) defined BETWEEN Relu and the depthwise Conv
// instead of at the top of the function. Guards the rewriter contract that
// the rewrite must be valid regardless of where constants are defined: the
// engine must place weight slices (and any other operand it fabricates) at a
// point that dominates every cloned use, hoisting constants when needed —
// otherwise this graph produces "operand does not dominate this use".
func.func @split_dominance(%arg0: tensor<1x8x16x16xf32>) -> tensor<1x16x8x8xf32> {
  %w1 = onnx.Constant dense<1.000000e-01> : tensor<16x8x3x3xf32>
  %b1 = onnx.Constant dense<2.000000e-01> : tensor<16xf32>
  %conv = "onnx.Conv"(%arg0, %w1, %b1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x8x16x16xf32>, tensor<16x8x3x3xf32>, tensor<16xf32>) -> tensor<1x16x16x16xf32>
  %relu = "onnx.Relu"(%conv) : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  %w2 = onnx.Constant dense<3.000000e-01> : tensor<16x1x3x3xf32>
  %b2 = onnx.Constant dense<4.000000e-01> : tensor<16xf32>
  %dw = "onnx.Conv"(%relu, %w2, %b2) {auto_pad = "NOTSET", dilations = [1, 1], group = 16 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x16x16x16xf32>, tensor<16x1x3x3xf32>, tensor<16xf32>) -> tensor<1x16x8x8xf32>
  %out = "onnx.Sigmoid"(%dw) : (tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
  return %out : tensor<1x16x8x8xf32>
}

// CHECK-LABEL: func.func @split_dominance
// CHECK: "onnx.Concat"({{.*}}, {{.*}}) {axis = 1 : si64} : (tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>) -> tensor<1x16x8x8xf32>
