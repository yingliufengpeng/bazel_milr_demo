// RUN: peng-opt %s --apply-distribute-transform  --split-input-file | FileCheck %s

module @Peng {
  // CHECK-LABEL: func @main(
  // CHECK-COUNT-4: peng.softmax
  func.func @main(%arg0: !peng.p_tensor<5x?x?xf32,0>) -> !peng.p_tensor<5x?x?xf32,0> attributes {dp_attr = #peng.DP<DP = 2 : 0, 1>, host_func} {
    %0 = "peng.softmax"(%arg0) <{axis = 1 : i64}> : (!peng.p_tensor<5x?x?xf32,0>) -> !peng.p_tensor<5x?x?xf32,0>
    %1 = "peng.softmax"(%0) <{axis = 1 : i64}> : (!peng.p_tensor<5x?x?xf32,0>) -> !peng.p_tensor<5x?x?xf32,0>
    return %1 : !peng.p_tensor<5x?x?xf32,0>
  }
}

// -----
module @Peng {
  // CHECK-LABEL: func @main(
  // CHECK-COUNT-3: peng.softmax
  // CHECK-NEXT: peng.buffer_cast
  // CHECK-NEXT: peng.buffer_cast
  // CHECK-COUNT-3: peng.softmax
  func.func @main(%arg0: !peng.p_tensor<5x?x?xf32,0>) -> !peng.p_tensor<5x?x?xf32,0> attributes {dp_attr = #peng.DP<DP = 3 : 0, 1, 2>, host_func} {
    %0 = "peng.softmax"(%arg0) <{axis = 1 : i64}> : (!peng.p_tensor<5x?x?xf32,0>) -> !peng.p_tensor<5x?x?xf32,0>
    %1 = "peng.softmax"(%0) <{axis = 1 : i64}> : (!peng.p_tensor<5x?x?xf32,0>) -> !peng.p_tensor<5x?x?xf32,0>
    return %1 : !peng.p_tensor<5x?x?xf32,0>
  }
}