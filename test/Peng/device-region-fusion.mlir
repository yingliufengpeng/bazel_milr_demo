// RUN: peng-opt %s  --device-region-fusion   --split-input-file | FileCheck %s

// CHECK-LABEL: peng
// CHECK: func.func
// CHECK-SAME: device_kernel
// CHECK-NEXT: peng.buffer_cast
// CHECK-NEXT: peng.softmax
// CHECK-NEXT: peng.softmax
// CHECK: func.func @main
// CHECK-COUNT-3: call
// CHECK-NOT: peng.softmax
module @Peng {
  func.func @main(%arg0: !peng.p_tensor<5x?x?xf32,0>) -> !peng.p_tensor<5x?x?xf32,0> attributes {dp_attr = #peng.DP<DP = 3 : 0, 1, 2>, host_func, device_kernel} {
    %0:3 = "peng.buffer_cast"(%arg0) <{distribute_attr = #peng.DP<DP = 3 : 0, 1, 2>}> : (!peng.p_tensor<5x?x?xf32,0>) -> (!peng.p_tensor<1x?x?xf32,0>, !peng.p_tensor<2x?x?xf32,1>, !peng.p_tensor<2x?x?xf32,2>)
    %1 = "peng.softmax"(%0#0) <{axis = 1 : i64}> : (!peng.p_tensor<1x?x?xf32,0>) -> !peng.p_tensor<1x?x?xf32,0>
    %2 = "peng.softmax"(%0#1) <{axis = 1 : i64}> : (!peng.p_tensor<2x?x?xf32,1>) -> !peng.p_tensor<2x?x?xf32,1>
    %3 = "peng.softmax"(%0#2) <{axis = 1 : i64}> : (!peng.p_tensor<2x?x?xf32,2>) -> !peng.p_tensor<2x?x?xf32,2>
    %4 = "peng.buffer_cast"(%1, %2, %3) <{distribute_attr = #peng.DP<DP = 3 : 0, 1, 2>}> : (!peng.p_tensor<1x?x?xf32,0>, !peng.p_tensor<2x?x?xf32,1>, !peng.p_tensor<2x?x?xf32,2>) -> !peng.p_tensor<5x?x?xf32,0>
    %5:3 = "peng.buffer_cast"(%4) <{distribute_attr = #peng.DP<DP = 3 : 0, 1, 2>}> : (!peng.p_tensor<5x?x?xf32,0>) -> (!peng.p_tensor<1x?x?xf32,0>, !peng.p_tensor<2x?x?xf32,1>, !peng.p_tensor<2x?x?xf32,2>)
    %6 = "peng.softmax"(%5#0) <{axis = 1 : i64}> : (!peng.p_tensor<1x?x?xf32,0>) -> !peng.p_tensor<1x?x?xf32,0>
    %7 = "peng.softmax"(%5#1) <{axis = 1 : i64}> : (!peng.p_tensor<2x?x?xf32,1>) -> !peng.p_tensor<2x?x?xf32,1>
    %8 = "peng.softmax"(%5#2) <{axis = 1 : i64}> : (!peng.p_tensor<2x?x?xf32,2>) -> !peng.p_tensor<2x?x?xf32,2>
    %9 = "peng.buffer_cast"(%6, %7, %8) <{distribute_attr = #peng.DP<DP = 3 : 0, 1, 2>}> : (!peng.p_tensor<1x?x?xf32,0>, !peng.p_tensor<2x?x?xf32,1>, !peng.p_tensor<2x?x?xf32,2>) -> !peng.p_tensor<5x?x?xf32,0>
    return %9 : !peng.p_tensor<5x?x?xf32,0>
  }
}