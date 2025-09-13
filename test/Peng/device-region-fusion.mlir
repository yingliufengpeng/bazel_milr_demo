// RUN: peng-opt %s  --device-region-fusion   --split-input-file | FileCheck %s

// CHECK-LABEL: Peng
// CHECK: func.func
// CHECK-SAME: device_kernel
// CHECK-NEXT: Peng.buffer_cast
// CHECK-NEXT: Peng.softmax
// CHECK-NEXT: Peng.softmax
// CHECK: func.func @main
// CHECK-COUNT-3: call
// CHECK-NOT: Peng.softmax
module @Peng {
  func.func @main(%arg0: !Peng.p_tensor<5x?x?xf32,0>) -> !Peng.p_tensor<5x?x?xf32,0> attributes {dp_attr = #Peng.DP<DP = 3 : 0, 1, 2>, host_func, device_kernel} {
    %0:3 = "Peng.buffer_cast"(%arg0) <{distribute_attr = #Peng.DP<DP = 3 : 0, 1, 2>}> : (!Peng.p_tensor<5x?x?xf32,0>) -> (!Peng.p_tensor<1x?x?xf32,0>, !Peng.p_tensor<2x?x?xf32,1>, !Peng.p_tensor<2x?x?xf32,2>)
    %1 = "Peng.softmax"(%0#0) <{axis = 1 : i64}> : (!Peng.p_tensor<1x?x?xf32,0>) -> !Peng.p_tensor<1x?x?xf32,0>
    %2 = "Peng.softmax"(%0#1) <{axis = 1 : i64}> : (!Peng.p_tensor<2x?x?xf32,1>) -> !Peng.p_tensor<2x?x?xf32,1>
    %3 = "Peng.softmax"(%0#2) <{axis = 1 : i64}> : (!Peng.p_tensor<2x?x?xf32,2>) -> !Peng.p_tensor<2x?x?xf32,2>
    %4 = "Peng.buffer_cast"(%1, %2, %3) <{distribute_attr = #Peng.DP<DP = 3 : 0, 1, 2>}> : (!Peng.p_tensor<1x?x?xf32,0>, !Peng.p_tensor<2x?x?xf32,1>, !Peng.p_tensor<2x?x?xf32,2>) -> !Peng.p_tensor<5x?x?xf32,0>
    %5:3 = "Peng.buffer_cast"(%4) <{distribute_attr = #Peng.DP<DP = 3 : 0, 1, 2>}> : (!Peng.p_tensor<5x?x?xf32,0>) -> (!Peng.p_tensor<1x?x?xf32,0>, !Peng.p_tensor<2x?x?xf32,1>, !Peng.p_tensor<2x?x?xf32,2>)
    %6 = "Peng.softmax"(%5#0) <{axis = 1 : i64}> : (!Peng.p_tensor<1x?x?xf32,0>) -> !Peng.p_tensor<1x?x?xf32,0>
    %7 = "Peng.softmax"(%5#1) <{axis = 1 : i64}> : (!Peng.p_tensor<2x?x?xf32,1>) -> !Peng.p_tensor<2x?x?xf32,1>
    %8 = "Peng.softmax"(%5#2) <{axis = 1 : i64}> : (!Peng.p_tensor<2x?x?xf32,2>) -> !Peng.p_tensor<2x?x?xf32,2>
    %9 = "Peng.buffer_cast"(%6, %7, %8) <{distribute_attr = #Peng.DP<DP = 3 : 0, 1, 2>}> : (!Peng.p_tensor<1x?x?xf32,0>, !Peng.p_tensor<2x?x?xf32,1>, !Peng.p_tensor<2x?x?xf32,2>) -> !Peng.p_tensor<5x?x?xf32,0>
    return %9 : !Peng.p_tensor<5x?x?xf32,0>
  }
}