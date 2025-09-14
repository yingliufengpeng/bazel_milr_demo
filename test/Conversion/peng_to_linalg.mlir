// RUN: peng-opt %s --convert-peng-to-linalg  --reconcile-unrealized-casts --split-input-file | FileCheck %s
module @Peng {
  // CHECK-COUNT-2: Peng.softmax
  // CHECK-COUNT-4: linalg.softmax
  func.func @main(%arg0: !Peng.p_tensor<5x?x?xf32,0>) -> !Peng.p_tensor<5x?x?xf32,0> attributes {dp_attr = #Peng.DP<DP = 3 : 0, 1, 2>, host_func} {
    %0:3 = "Peng.buffer_cast"(%arg0) <{distribute_attr = #Peng.DP<DP = 3 : 0, 1, 2>}> : (!Peng.p_tensor<5x?x?xf32,0>) -> (!Peng.p_tensor<1x?x?xf32,0>, !Peng.p_tensor<2x?x?xf32,1>, !Peng.p_tensor<2x?x?xf32,2>)
    %1 = "Peng.softmax"(%0#0) <{axis = 1 : i64}> : (!Peng.p_tensor<1x?x?xf32,0>) -> !Peng.p_tensor<1x?x?xf32,0>
    %6 = "Peng.softmax"(%1) <{axis = 1 : i64}> : (!Peng.p_tensor<1x?x?xf32,0>) -> !Peng.p_tensor<1x?x?xf32,0>
    %2 = "Peng.device_region"(%0#1) <{device_id = 1 : i64, sym_name = "softmax_2_d_d_softmax_2_d_d_"}> ({
    ^bb0(%arg1: !Peng.p_tensor<2x?x?xf32,1>):
      %52 = "Peng.softmax"(%arg1) <{axis = 1 : i64}> : (!Peng.p_tensor<2x?x?xf32,1>) -> !Peng.p_tensor<2x?x?xf32,1>
      %62 = "Peng.softmax"(%52) <{axis = 1 : i64}> : (!Peng.p_tensor<2x?x?xf32,1>) -> !Peng.p_tensor<2x?x?xf32,1>
      Peng.return %62 : !Peng.p_tensor<2x?x?xf32,1>
    }) : (!Peng.p_tensor<2x?x?xf32,1>) -> !Peng.p_tensor<2x?x?xf32,1>
    %3 = "Peng.device_region"(%0#2) <{device_id = 2 : i64, sym_name = "softmax_2_d_d_softmax_2_d_d_"}> ({
    ^bb0(%arg1: !Peng.p_tensor<2x?x?xf32,2>):
      %53 = "Peng.softmax"(%arg1) <{axis = 1 : i64}> : (!Peng.p_tensor<2x?x?xf32,2>) -> !Peng.p_tensor<2x?x?xf32,2>
      %63 = "Peng.softmax"(%53) <{axis = 1 : i64}> : (!Peng.p_tensor<2x?x?xf32,2>) -> !Peng.p_tensor<2x?x?xf32,2>
      Peng.return %63 : !Peng.p_tensor<2x?x?xf32,2>
    }) : (!Peng.p_tensor<2x?x?xf32,2>) -> !Peng.p_tensor<2x?x?xf32,2>
    %4 = "Peng.buffer_cast"(%6, %2, %3) <{distribute_attr = #Peng.DP<DP = 3 : 0, 1, 2>}> : (!Peng.p_tensor<1x?x?xf32,0>, !Peng.p_tensor<2x?x?xf32,1>, !Peng.p_tensor<2x?x?xf32,2>) -> !Peng.p_tensor<5x?x?xf32,0>
    return %4 : !Peng.p_tensor<5x?x?xf32,0>
  }
}