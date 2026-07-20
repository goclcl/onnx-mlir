// RUN: onnx-mlir-opt --onnx-peak-mem-opt --shape-inference %s | FileCheck %s

// Expand/Shrink pattern, Conv channel split (mobilenetv2-style).
// S = expanding Conv (8ch -> 16ch), in-path Relu, E = shrinking depthwise
// Conv (stride 2). Peak is at Relu (conv_out + relu_out live).
// Expected rewrite: S weights/bias split along output channel into two Convs,
// path cloned per half (depthwise Conv gets group halved), merged with
// Concat(axis=1). Trailing Sigmoid consumes E's result from outside.
// NOTE: constants are hoisted to the top on purpose — the current rewriter
// inserts weight Splits right after the constant, so constants defined after
// S would not dominate the cloned path (see the *_split_dominance test).
func.func @expand_shrink_conv(%arg0: tensor<1x8x16x16xf32>) -> tensor<1x16x8x8xf32> {
  %w1 = onnx.Constant dense<1.000000e-01> : tensor<16x8x3x3xf32>
  %b1 = onnx.Constant dense<2.000000e-01> : tensor<16xf32>
  %w2 = onnx.Constant dense<3.000000e-01> : tensor<16x1x3x3xf32>
  %b2 = onnx.Constant dense<4.000000e-01> : tensor<16xf32>
  %conv = "onnx.Conv"(%arg0, %w1, %b1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x8x16x16xf32>, tensor<16x8x3x3xf32>, tensor<16xf32>) -> tensor<1x16x16x16xf32>
  %relu = "onnx.Relu"(%conv) : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  %dw = "onnx.Conv"(%relu, %w2, %b2) {auto_pad = "NOTSET", dilations = [1, 1], group = 16 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x16x16x16xf32>, tensor<16x1x3x3xf32>, tensor<16xf32>) -> tensor<1x16x8x8xf32>
  %out = "onnx.Sigmoid"(%dw) : (tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
  return %out : tensor<1x16x8x8xf32>
}

// CHECK-LABEL: func.func @expand_shrink_conv
// CHECK: [[W1S:%.+]]:2 = "onnx.Split"({{.*}}) {axis = 0 : si64, num_outputs = 2 : si64} : (tensor<16x8x3x3xf32>, none) -> (tensor<8x8x3x3xf32>, tensor<8x8x3x3xf32>)
// CHECK: [[B1S:%.+]]:2 = "onnx.Split"({{.*}}) {axis = 0 : si64, num_outputs = 2 : si64} : (tensor<16xf32>, none) -> (tensor<8xf32>, tensor<8xf32>)
// CHECK: [[CONV_A:%.+]] = "onnx.Conv"(%arg0, [[W1S]]#0, [[B1S]]#0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x8x16x16xf32>, tensor<8x8x3x3xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
// CHECK: [[RELU_A:%.+]] = "onnx.Relu"([[CONV_A]]) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
// CHECK: [[DW_A:%.+]] = "onnx.Conv"([[RELU_A]], {{.*}}, {{.*}}) {auto_pad = "NOTSET", dilations = [1, 1], group = 8 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x8x16x16xf32>, tensor<8x1x3x3xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
// CHECK: [[CONV_B:%.+]] = "onnx.Conv"(%arg0, [[W1S]]#1, [[B1S]]#1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x8x16x16xf32>, tensor<8x8x3x3xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
// CHECK: [[RELU_B:%.+]] = "onnx.Relu"([[CONV_B]]) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
// CHECK: [[DW_B:%.+]] = "onnx.Conv"([[RELU_B]], {{.*}}, {{.*}}) {auto_pad = "NOTSET", dilations = [1, 1], group = 8 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x8x16x16xf32>, tensor<8x1x3x3xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
// CHECK: [[CAT:%.+]] = "onnx.Concat"([[DW_A]], [[DW_B]]) {axis = 1 : si64} : (tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>) -> tensor<1x16x8x8xf32>
// CHECK: [[OUT:%.+]] = "onnx.Sigmoid"([[CAT]]) : (tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
// CHECK: return [[OUT]] : tensor<1x16x8x8xf32>
