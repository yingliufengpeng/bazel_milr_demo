// RUN: peng-opt %s   --split-input-file --inline | FileCheck %s

// CHECK-LABEL: peng
// CHECK-NOT: peng.add
module @Peng {
  func.func @main() -> !peng.p_tensor<2x2xf32,0> {
    %0 = "peng.const"() <{value = dense<[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]> : !peng.p_tensor<2x2xf32,1>}> : () -> !peng.p_tensor<2x2xf32,0>
    %1 = "peng.const"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : !peng.p_tensor<2x2xf32,1>}> : () -> !peng.p_tensor<2x2xf32,0>
    %2 = "peng.add"(%0, %1) : (!peng.p_tensor<2x2xf32,0>, !peng.p_tensor<2x2xf32,0>) -> !peng.p_tensor<2x2xf32,0>
    return %2 : !peng.p_tensor<2x2xf32,0>
  }
}