// RUN: onnx-mlir-opt --onnx-peak-mem-opt --shape-inference %s | FileCheck %s

// Fork/Join pattern: SiLU (x * sigmoid(x)), the efficientnetv2/yolov10s
// case. The Conv output forks into Sigmoid and Mul; the branches rejoin at
// Mul (E). The expand/shrink matcher only accepts single chains — a fork
// producer cannot head a chain, so its search skips past the Conv and finds
// NO match here; the fork/join matcher claims the region instead. Verifies
// the path cloner handles a DAG (not just a chain): each half must get its
// own Sigmoid AND Mul wired to the same half-Conv, merged with
// Concat(axis=1).
func.func @fork_join_silu(%arg0: tensor<1x8x16x16xf32>) -> tensor<1x16x16x16xf32> {
  %w = onnx.Constant dense<1.000000e-01> : tensor<16x8x3x3xf32>
  %b = onnx.Constant dense<2.000000e-01> : tensor<16xf32>
  %conv = "onnx.Conv"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x8x16x16xf32>, tensor<16x8x3x3xf32>, tensor<16xf32>) -> tensor<1x16x16x16xf32>
  %sig = "onnx.Sigmoid"(%conv) : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  %mul = "onnx.Mul"(%conv, %sig) : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  %out = "onnx.Relu"(%mul) : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  return %out : tensor<1x16x16x16xf32>
}

// CHECK-LABEL: func.func @fork_join_silu
// CHECK: [[WS:%.+]]:2 = "onnx.Split"({{.*}}) {axis = 0 : si64, num_outputs = 2 : si64} : (tensor<16x8x3x3xf32>, none) -> (tensor<8x8x3x3xf32>, tensor<8x8x3x3xf32>)
// CHECK: [[BS:%.+]]:2 = "onnx.Split"({{.*}}) {axis = 0 : si64, num_outputs = 2 : si64} : (tensor<16xf32>, none) -> (tensor<8xf32>, tensor<8xf32>)
// CHECK: [[CONV_A:%.+]] = "onnx.Conv"(%arg0, [[WS]]#0, [[BS]]#0) {{.*}} -> tensor<1x8x16x16xf32>
// Both fork users must be wired to the SAME half-Conv within each path:
// CHECK: [[SIG_A:%.+]] = "onnx.Sigmoid"([[CONV_A]]) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
// CHECK: [[MUL_A:%.+]] = "onnx.Mul"([[CONV_A]], [[SIG_A]]) : (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
// CHECK: [[CONV_B:%.+]] = "onnx.Conv"(%arg0, [[WS]]#1, [[BS]]#1) {{.*}} -> tensor<1x8x16x16xf32>
// CHECK: [[SIG_B:%.+]] = "onnx.Sigmoid"([[CONV_B]]) : (tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
// CHECK: [[MUL_B:%.+]] = "onnx.Mul"([[CONV_B]], [[SIG_B]]) : (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
// CHECK: [[CAT:%.+]] = "onnx.Concat"([[MUL_A]], [[MUL_B]]) {axis = 1 : si64} : (tensor<1x8x16x16xf32>, tensor<1x8x16x16xf32>) -> tensor<1x16x16x16xf32>
// CHECK: [[OUT:%.+]] = "onnx.Relu"([[CAT]]) : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
// CHECK: return [[OUT]] : tensor<1x16x16x16xf32>
