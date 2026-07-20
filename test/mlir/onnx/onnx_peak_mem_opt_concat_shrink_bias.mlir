// RUN: onnx-mlir-opt --onnx-peak-mem-opt --shape-inference %s | FileCheck %s

// Concat-Shrink pattern with a biased Conv.
// Concat(axis=1) of two 8ch tensors feeds a Conv (16ch -> 8ch, WITH bias).
// Expected rewrite: drop the Concat, run one Conv per branch on the matching
// input-channel slice of the weight, and merge the partial results with
// Sum/Add. The none-bias CHECKs are the regression guard for the invariant
// that an additive operand under a partial-sum merge is applied EXACTLY
// ONCE: branch 0 keeps it, every other branch gets none — applying it per
// branch would add it N times into the merged result.
func.func @concat_shrink_bias(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32> {
  %w = onnx.Constant dense<1.000000e-01> : tensor<8x16x3x3xf32>
  %bias = onnx.Constant dense<2.000000e-01> : tensor<8xf32>
  %a = "onnx.Relu"(%arg0) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
  %b = "onnx.Sigmoid"(%arg1) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
  %cat = "onnx.Concat"(%a, %b) {axis = 1 : si64} : (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) -> tensor<1x16x16x16xf32>
  %conv = "onnx.Conv"(%cat, %w, %bias) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x16x16x16xf32>, tensor<8x16x3x3xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
  %out = "onnx.Relu"(%conv) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
  return %out : tensor<1x8x16x16xf32>
}

// CHECK-LABEL: func.func @concat_shrink_bias
// CHECK: [[A:%.+]] = "onnx.Relu"(%arg0)
// CHECK: [[B:%.+]] = "onnx.Sigmoid"(%arg1)
// Exactly one branch keeps the bias (type tensor<8xf32>), the other gets none:
// CHECK-DAG: "onnx.Conv"([[A]], {{.*}}) {{.*}} : (tensor<1x8x16x16xf32>, tensor<8x8x3x3xf32>, tensor<8xf32>) -> tensor<1x8x16x16xf32>
// CHECK-DAG: "onnx.Conv"([[B]], {{.*}}) {{.*}} : (tensor<1x8x16x16xf32>, tensor<8x8x3x3xf32>, none) -> tensor<1x8x16x16xf32>
// The partial results are merged with Sum/Add (input-channel split reduces):
// CHECK: [[MERGE:%.+]] = "onnx.{{(Sum|Add)}}"({{.*}}, {{.*}}) : (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
// CHECK: [[OUT:%.+]] = "onnx.Relu"([[MERGE]]) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
// CHECK: return [[OUT]] : tensor<1x8x16x16xf32>
