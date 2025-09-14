// RUN: peng-opt %s --convert-peng-to-linalg  --reconcile-unrealized-casts --split-input-file | FileCheck %s
module @Peng {
  // CHECK-COUNT-2: peng.softmax
  // CHECK-COUNT-4: linalg.softmax
  func.func @main(%arg0: !peng.p_tensor<5x?x?xf32,0>) -> !peng.p_tensor<5x?x?xf32,0> attributes {dp_attr = #peng.DP<DP = 3 : 0, 1, 2>, host_func} {
    %0:3 = "peng.buffer_cast"(%arg0) <{distribute_attr = #peng.DP<DP = 3 : 0, 1, 2>}> : (!peng.p_tensor<5x?x?xf32,0>) -> (!peng.p_tensor<1x?x?xf32,0>, !peng.p_tensor<2x?x?xf32,1>, !peng.p_tensor<2x?x?xf32,2>)
    %1 = "peng.softmax"(%0#0) <{axis = 1 : i64}> : (!peng.p_tensor<1x?x?xf32,0>) -> !peng.p_tensor<1x?x?xf32,0>
    %6 = "peng.softmax"(%1) <{axis = 1 : i64}> : (!peng.p_tensor<1x?x?xf32,0>) -> !peng.p_tensor<1x?x?xf32,0>
    %2 = "peng.device_region"(%0#1) <{device_id = 1 : i64, sym_name = "softmax_2_d_d_softmax_2_d_d_"}> ({
    ^bb0(%arg1: !peng.p_tensor<2x?x?xf32,1>):
      %52 = "peng.softmax"(%arg1) <{axis = 1 : i64}> : (!peng.p_tensor<2x?x?xf32,1>) -> !peng.p_tensor<2x?x?xf32,1>
      %62 = "peng.softmax"(%52) <{axis = 1 : i64}> : (!peng.p_tensor<2x?x?xf32,1>) -> !peng.p_tensor<2x?x?xf32,1>
      peng.return %62 : !peng.p_tensor<2x?x?xf32,1>
    }) : (!peng.p_tensor<2x?x?xf32,1>) -> !peng.p_tensor<2x?x?xf32,1>
    %3 = "peng.device_region"(%0#2) <{device_id = 2 : i64, sym_name = "softmax_2_d_d_softmax_2_d_d_"}> ({
    ^bb0(%arg1: !peng.p_tensor<2x?x?xf32,2>):
      %53 = "peng.softmax"(%arg1) <{axis = 1 : i64}> : (!peng.p_tensor<2x?x?xf32,2>) -> !peng.p_tensor<2x?x?xf32,2>
      %63 = "peng.softmax"(%53) <{axis = 1 : i64}> : (!peng.p_tensor<2x?x?xf32,2>) -> !peng.p_tensor<2x?x?xf32,2>
      peng.return %63 : !peng.p_tensor<2x?x?xf32,2>
    }) : (!peng.p_tensor<2x?x?xf32,2>) -> !peng.p_tensor<2x?x?xf32,2>
    %4 = "peng.buffer_cast"(%6, %2, %3) <{distribute_attr = #peng.DP<DP = 3 : 0, 1, 2>}> : (!peng.p_tensor<1x?x?xf32,0>, !peng.p_tensor<2x?x?xf32,1>, !peng.p_tensor<2x?x?xf32,2>) -> !peng.p_tensor<5x?x?xf32,0>
    return %4 : !peng.p_tensor<5x?x?xf32,0>
  }
}