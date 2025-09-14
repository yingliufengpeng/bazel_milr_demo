#include <iostream>
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "include/PengDialect.h"
#include "include/PengTypes.h"
#include "include/PengAttrs.h"
#include "include/PengEnums.h"
#include "include/PengAttrs.h"
#include "include/Transforms/Passes.h"
#include "include/PengOps.h"
#include "include/DistributeParallelismInterfaces.h"
#include "include/Utils/File.h"
#include "include/Utils/Key.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
namespace {

} // namespace


void CH14() {
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  context.disableMultithreading(true);
  // 加载/注册方言
  context.getOrLoadDialect<mlir::peng::PengDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto type = mlir::peng::PTensorType::get(&context, {2, 2},
                                                  builder.getF32Type(), 1);
  auto const_op = builder.create<mlir::peng::ConstOp>(
      loc, type, mlir::DenseElementsAttr::get(type, mlir::ArrayRef<float>{1.0, 2., 3., 4.}));
  const_op->dump();
}

// 1.规范化定义：   let hasCanonicalizeMethod = 1;    // Op 生成一个重写的函数
//                 let hasCanonicalizer = 1;         // Op
//                 生成多个规范化的Pattern的函数
//   定义一些比较通用的pattern，在运行Pass/Pattern后自动触发，使IR更加简介且避免Pipeline过于臃肿。

// 2. 常量折叠：    let hasFolder = 1;                // Op 生成 fold 函数
//                 let hasConstantMaterializer = 1;  // Dialect 生成的Const
//                 Operation 的函数

int main() { CH14(); }
