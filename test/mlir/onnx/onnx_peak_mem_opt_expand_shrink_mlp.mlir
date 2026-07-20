// RUN: onnx-mlir-opt --onnx-peak-mem-opt --shape-inference %s | FileCheck %s

// Expand/Shrink pattern, MLP hidden-dim (K) split (vit/convnext-style).
// S = fc1 MatMul (8 -> 64 hidden), in-path bias Add + Gelu, E = fc2 MatMul
// (64 -> 8). split-dim=2 is the hidden dim, which is the reduction (K) dim of
// the E MatMul, so the two partial results must be merged with Add (not
// Concat). fc1 weight is split along N, bias along its only dim, fc2 weight
// along K.
func.func @expand_shrink_mlp(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %w1 = onnx.Constant dense<1.000000e-01> : tensor<8x64xf32>
  %b1 = onnx.Constant dense<2.000000e-01> : tensor<64xf32>
  %w2 = onnx.Constant dense<3.000000e-01> : tensor<64x8xf32>
  %mm1 = "onnx.MatMul"(%arg0, %w1) : (tensor<1x4x8xf32>, tensor<8x64xf32>) -> tensor<1x4x64xf32>
  %add = "onnx.Add"(%mm1, %b1) : (tensor<1x4x64xf32>, tensor<64xf32>) -> tensor<1x4x64xf32>
  %gelu = "onnx.Gelu"(%add) {approximate = "none"} : (tensor<1x4x64xf32>) -> tensor<1x4x64xf32>
  %mm2 = "onnx.MatMul"(%gelu, %w2) : (tensor<1x4x64xf32>, tensor<64x8xf32>) -> tensor<1x4x8xf32>
  %out = "onnx.Relu"(%mm2) : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  return %out : tensor<1x4x8xf32>
}

// CHECK-LABEL: func.func @expand_shrink_mlp
// CHECK: [[W1S:%.+]]:2 = "onnx.Split"({{.*}}) {axis = 1 : si64, num_outputs = 2 : si64} : (tensor<8x64xf32>, none) -> (tensor<8x32xf32>, tensor<8x32xf32>)
// CHECK: [[MM1_A:%.+]] = "onnx.MatMul"(%arg0, [[W1S]]#0) : (tensor<1x4x8xf32>, tensor<8x32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[ADD_A:%.+]] = "onnx.Add"([[MM1_A]], {{.*}}) : (tensor<1x4x32xf32>, tensor<32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[GELU_A:%.+]] = "onnx.Gelu"([[ADD_A]]) {approximate = "none"} : (tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[MM2_A:%.+]] = "onnx.MatMul"([[GELU_A]], {{.*}}) : (tensor<1x4x32xf32>, tensor<32x8xf32>) -> tensor<1x4x8xf32>
// CHECK: [[MM1_B:%.+]] = "onnx.MatMul"(%arg0, [[W1S]]#1) : (tensor<1x4x8xf32>, tensor<8x32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[ADD_B:%.+]] = "onnx.Add"([[MM1_B]], {{.*}}) : (tensor<1x4x32xf32>, tensor<32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[GELU_B:%.+]] = "onnx.Gelu"([[ADD_B]]) {approximate = "none"} : (tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
// CHECK: [[MM2_B:%.+]] = "onnx.MatMul"([[GELU_B]], {{.*}}) : (tensor<1x4x32xf32>, tensor<32x8xf32>) -> tensor<1x4x8xf32>
// The K-dim partial results must be merged with Add, not Concat:
// CHECK: [[MERGE:%.+]] = "onnx.Add"([[MM2_A]], [[MM2_B]]) : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
// CHECK: [[OUT:%.+]] = "onnx.Relu"([[MERGE]]) : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
// CHECK: return [[OUT]] : tensor<1x4x8xf32>
