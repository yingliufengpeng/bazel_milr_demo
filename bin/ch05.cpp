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


void CH5() {
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  context.getOrLoadDialect<mlir::peng::PengDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  // ModuleOp
  auto module = builder.create<mlir::ModuleOp>(loc, "Peng");
  builder.setInsertionPointToStart(module.getBody());
  // ConstOp
  auto f32 = mlir::Float32Type::get(&context);
  auto shape = mlir::SmallVector<int64_t>({2, 2});
  auto const_value_1 =
      mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)1));
  auto const_value_2 =
      mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)2));
  auto tensor_type_1 =
      mlir::peng::PTensorType::get(&context, shape, f32, 0);
  auto tensor_type_2 =
      mlir::peng::PTensorType::get(&context, shape, f32, 1);
  auto const_1 = builder.create<mlir::peng::ConstOp>(
      loc, tensor_type_1,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_1));
  auto const_2 = builder.create<mlir::peng::ConstOp>(
      loc, tensor_type_1,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_1));
  auto const_3 = builder.create<mlir::peng::ConstOp>(
      loc, tensor_type_2,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_2));
  auto const_4 = builder.create<mlir::peng::ConstOp>(
      loc, tensor_type_2,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_2));
  llvm::outs() << "Const tensor in divece 0 :\n";
  const_1->dump();
  llvm::outs() << "Const tensor in divece 1 :\n";
  const_3->dump();
  // Buffer Op
  auto buffer_op = builder.create<mlir::peng::BufferOp>(
      loc, mlir::ValueRange({const_1, const_3}));
  llvm::outs() << "Buffer Op :\n";
  buffer_op->dump();
  // Get Tensor Op
  auto get_tensor_op_1 = builder.create<mlir::peng::GetTensorOp>(
      loc, tensor_type_1, buffer_op, 0);
  auto get_tensor_op_2 = builder.create<mlir::peng::GetTensorOp>(
      loc, tensor_type_2, buffer_op, 1);
  llvm::outs() << "Get Tensor Op :\n";
  get_tensor_op_1->dump();
  get_tensor_op_2->dump();
  // Softmax Op
  auto softmax_op =
      builder.create<mlir::peng::SoftmaxOp>(loc, get_tensor_op_1, 1);
  llvm::outs() << "Softmax Op :\n";
  softmax_op->dump();
  // Exp Op
  auto exp_op = builder.create<mlir::peng::ExpOp>(loc, get_tensor_op_2);
  llvm::outs() << "Exp Op :\n";
  exp_op->dump();
  // all to all op
  auto out_buffer_op = builder.create<mlir::peng::BufferOp>(
      loc, mlir::ValueRange({const_2, const_4}));
  auto all_to_all_op = builder.create<mlir::peng::AllToAllOp>(
      loc, buffer_op, out_buffer_op);
  llvm::outs() << "All to All Op :\n";
  all_to_all_op->dump();
}


int main() { CH5(); }
