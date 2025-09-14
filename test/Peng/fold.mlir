// RUN: peng-opt %s   --split-input-file --inline | FileCheck %s

// CHECK-LABEL: Peng
// CHECK-NOT: Peng.add
module @Peng {
  func.func @main() -> !Peng.p_tensor<2x2xf32,0> {
    %0 = "Peng.const"() <{value = dense<[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]> : !Peng.p_tensor<2x2xf32,1>}> : () -> !Peng.p_tensor<2x2xf32,0>
    %1 = "Peng.const"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : !Peng.p_tensor<2x2xf32,1>}> : () -> !Peng.p_tensor<2x2xf32,0>
    %2 = "Peng.add"(%0, %1) : (!Peng.p_tensor<2x2xf32,0>, !Peng.p_tensor<2x2xf32,0>) -> !Peng.p_tensor<2x2xf32,0>
    return %2 : !Peng.p_tensor<2x2xf32,0>
  }
}