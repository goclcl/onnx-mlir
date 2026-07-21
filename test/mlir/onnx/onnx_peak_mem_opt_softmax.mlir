// RUN: onnx-mlir-opt --onnx-peak-mem-opt --shape-inference %s | FileCheck %s

// Expand/Shrink pattern through Softmax (vit attention-style).
// S = QK MatMul (dk=4 -> T=8, scores 1x2x8x8), Softmax(axis=-1), E = AV MatMul
// (T=8 -> dk=4). Softmax normalizes only its own axis (-1 == 3), so any other
// partition axis passes through unchanged. The weight-split option (Wk along N)
// would partition the softmax axis itself and must be rejected; the planner
// falls back to input-split. Head axis (1) and query axis (2) tie on balance
// (both split exactly in half), so the outer one (head) wins. rank is
// preserved, so the negative axis attr stays valid on the clones.
func.func @attention_softmax(%arg0: tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32> {
  %wk = onnx.Constant dense<1.000000e-01> : tensor<4x8xf32>
  %wv = onnx.Constant dense<2.000000e-01> : tensor<8x4xf32>
  %qk = "onnx.MatMul"(%arg0, %wk) : (tensor<1x2x8x4xf32>, tensor<4x8xf32>) -> tensor<1x2x8x8xf32>
  %sm = "onnx.Softmax"(%qk) {axis = -1 : si64} : (tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
  %av = "onnx.MatMul"(%sm, %wv) : (tensor<1x2x8x8xf32>, tensor<8x4xf32>) -> tensor<1x2x8x4xf32>
  %out = "onnx.Relu"(%av) : (tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  return %out : tensor<1x2x8x4xf32>
}

// CHECK-LABEL: func.func @attention_softmax
// CHECK: [[XS:%.+]]:2 = "onnx.Split"(%arg0, {{.*}}) {axis = 1 : si64, num_outputs = 2 : si64} : (tensor<1x2x8x4xf32>, none) -> (tensor<1x1x8x4xf32>, tensor<1x1x8x4xf32>)
// CHECK: [[QK_A:%.+]] = "onnx.MatMul"([[XS]]#0, {{.*}}) : (tensor<1x1x8x4xf32>, tensor<4x8xf32>) -> tensor<1x1x8x8xf32>
// CHECK: [[SM_A:%.+]] = "onnx.Softmax"([[QK_A]]) {axis = -1 : si64} : (tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>
// CHECK: [[AV_A:%.+]] = "onnx.MatMul"([[SM_A]], {{.*}}) : (tensor<1x1x8x8xf32>, tensor<8x4xf32>) -> tensor<1x1x8x4xf32>
// CHECK: [[QK_B:%.+]] = "onnx.MatMul"([[XS]]#1, {{.*}}) : (tensor<1x1x8x4xf32>, tensor<4x8xf32>) -> tensor<1x1x8x8xf32>
// CHECK: [[SM_B:%.+]] = "onnx.Softmax"([[QK_B]]) {axis = -1 : si64} : (tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>
// CHECK: [[AV_B:%.+]] = "onnx.MatMul"([[SM_B]], {{.*}}) : (tensor<1x1x8x8xf32>, tensor<8x4xf32>) -> tensor<1x1x8x4xf32>
// The partition axis survives to E, so the merge is Concat on that axis:
// CHECK: [[MERGE:%.+]] = "onnx.Concat"([[AV_A]], [[AV_B]]) {axis = 1 : si64} : (tensor<1x1x8x4xf32>, tensor<1x1x8x4xf32>) -> tensor<1x2x8x4xf32>
// CHECK: [[OUT:%.+]] = "onnx.Relu"([[MERGE]]) : (tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
// CHECK: return [[OUT]] : tensor<1x2x8x4xf32>
