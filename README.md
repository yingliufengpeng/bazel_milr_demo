
# 构建指令
    https://github.com/yingliufengpeng/bazel_project2.git
    构建指令位于README.md中

# 参考资料
    https://github.com/violetDelia/MLIR-Tutorial.git
    https://github.com/j2kun/mlir-tutorial.git

# 注意:  
    目前与test相关的测试逻辑，并不能在windows上运行成功。 原因在于编译期，重复的枚举定义。
# learning
    此代码的目的是为了学习 在llvm mlir中 集成bazel的相关知识,以及对mlir中相关模块的使用流程. 

# 常用的命令为
    bazelisk run //:bin/ch02.cpp_main
    bazelisk run //:bin/ch14.cpp_main
    bazelisk run     //:main
    bazelisk run     //:peng-opt -- tests/softmax.mlir
    bazelisk test    //tests:Conversion/peng_to_linalg.mlir.test
    bazelisk test    //tests:Peng/apply-distribute-transform.mlir.test
    bazelisk test    //tests:Peng/device-region-fusion.mlir.test
