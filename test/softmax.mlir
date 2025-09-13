
module @Peng {
  func.func @main(%arg0: !Peng.p_tensor<5x?x?xf32,0>) -> !Peng.p_tensor<5x?x?xf32,0> attributes {dp_attr = #Peng.DP<DP = 2 : 0, 1>, host_func} {
    %0 = "Peng.softmax"(%arg0) <{axis = 1 : i64}> : (!Peng.p_tensor<5x?x?xf32,0>) -> !Peng.p_tensor<5x?x?xf32,0>
    %1 = "Peng.softmax"(%0) <{axis = 1 : i64}> : (!Peng.p_tensor<5x?x?xf32,0>) -> !Peng.p_tensor<5x?x?xf32,0>
    return %1 : !Peng.p_tensor<5x?x?xf32,0>
  }
}