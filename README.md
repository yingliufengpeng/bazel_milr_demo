
构建指令
    https://github.com/yingliufengpeng/bazel_project2.git
    构建指令位于README.md中

参考资料
    https://github.com/violetDelia/MLIR-Tutorial.git
    https://github.com/j2kun/mlir-tutorial.git

此代码的目的是为了学习 在llvm mlir中 集成bazel的相关知识,以及对mlir中相关模块的使用流程. 

常用的命令为
bazelisk run     //:main
bazelisk run     //:peng-opt -- test/softmax.mlir
bazelisk test    //test:Conversion/peng_to_linalg.mlir.test
bazelisk test    //test:Peng/apply-distribute-transform.mlir.test
bazelisk test    //test:Peng/device-region-fusion.mlir.test
