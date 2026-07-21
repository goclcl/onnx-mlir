// RUN: onnx-mlir-opt --onnx-peak-mem-opt --shape-inference %s | FileCheck %s

// Expand/Shrink pattern through Softmax. S = fc1 MatMul (8 -> 64), Softmax
// over axis 1, E = fc2 MatMul (64 -> 8). The weight-split of W1 along N
// partitions axis 2, which lies AFTER the softmax axis — softmax normalizes
// only its own axis, so this passes (a LayerNormalization at the same spot
// would reject it: it normalizes the whole [axis, rank) suffix). The
// partition axis is the K dim of the E MatMul, so the partial results are
// merged with Add. rank is preserved, so the axis attr needs no patching.
func.func @softmax_passthrough(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %w1 = onnx.Constant dense<1.000000e-01> : tensor<8x64xf32>
  %w2 = onnx.Constant dense<3.000000e-01> : tensor<64x8xf32>
  %mm1 = "onnx.MatMul"(%arg0, %w1) : (tensor<1x4x8xf32>, tensor<8x64xf32>) -> tensor<1x4x64xf32>
  %sm = "onnx.Softmax"(%mm1) {axis = 1 : si64} : (tensor<1x4x64xf32>) -> tensor<1x4x64xf32>
  %mm2 = "onnx.MatMul"(%sm, %w2) : (tensor<1x4x64xf32>, tensor<64x8xf32>) -> tensor<1x4x8xf32>
  %out = "onnx.Relu"(%mm2) : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  return %out : tensor<1x4x8xf32>
}

// CHECK-LABEL: func.func @softmax_passthrough
// CHECK: [[W1S:%.+]]:2 = "onnx.Split"({{.*}}) {axis = 1 : si64, num_outputs = 2 : si64} : (tensor<8x64xf32>, none) -> (tensor<8x32xf32>, tensor<8x32xf32>)
// CHECK: [[MM1_A:%.+]] = "onnx.MatMul"(%arg0, [[W1S]]#0) : (tensor<1x4x8xf32>, tensor<8x32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[SM_A:%.+]] = "onnx.Softmax"([[MM1_A]]) {axis = 1 : si64} : (tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[MM2_A:%.+]] = "onnx.MatMul"([[SM_A]], {{.*}}) : (tensor<1x4x32xf32>, tensor<32x8xf32>) -> tensor<1x4x8xf32>
// CHECK: [[MM1_B:%.+]] = "onnx.MatMul"(%arg0, [[W1S]]#1) : (tensor<1x4x8xf32>, tensor<8x32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[SM_B:%.+]] = "onnx.Softmax"([[MM1_B]]) {axis = 1 : si64} : (tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[MM2_B:%.+]] = "onnx.MatMul"([[SM_B]], {{.*}}) : (tensor<1x4x32xf32>, tensor<32x8xf32>) -> tensor<1x4x8xf32>
// The K-dim partial results must be merged with Add, not Concat:
// CHECK: [[MERGE:%.+]] = "onnx.Add"([[MM2_A]], [[MM2_B]]) : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
// CHECK: [[OUT:%.+]] = "onnx.Relu"([[MERGE]]) : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
// CHECK: return [[OUT]] : tensor<1x4x8xf32>
