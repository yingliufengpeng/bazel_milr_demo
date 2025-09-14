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

void CH3() {
    mlir::DialectRegistry DialectRegistry;
    mlir::MLIRContext context(DialectRegistry);
    auto diaglect = context.getOrLoadDialect<mlir::peng::PengDialect>();

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();


    // ModuleOp
    auto module = builder.create<mlir::ModuleOp>(loc, "PengDialect");
    builder.setInsertionPointToStart(module.getBody());
    // ConstOp
    auto f32 = mlir::Float32Type::get(&context);
    auto shape = mlir::SmallVector<int64_t>({2, 2});
    auto const_value_1 =
            mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float) 1));
    auto const_value_2 =
            mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float) 2));
    auto tensor_type_1 =
            mlir::peng::PTensorType::get(&context, shape, f32, 0);
    auto tensor_type_2 =
            mlir::peng::PTensorType::get(&context, shape, f32, 1);
    auto const_1 = builder.create<mlir::peng::ConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_1));

    llvm::outs() << "const_1  " << const_1 << "\n";

    auto const_2 = builder.create<mlir::peng::ConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_1));

    llvm::outs() << "const_2  " << const_2 << "\n";


    auto const_3 = builder.create<mlir::peng::ConstOp>(
        loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_2));

    llvm::outs() << "const_3  " << const_3 << "\n";

    auto const_4 = builder.create<mlir::peng::ConstOp>(
        loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_2));

    llvm::outs() << "const_4  " << const_4 << "\n";

    // Buffer Op
    auto buffer_op = builder.create<mlir::peng::BufferOp>(
        loc, mlir::ValueRange({const_2, const_4,}));
    llvm::outs() << "buffer_op  " << buffer_op << "\n";


    // Get Tensor Op
    auto get_tensor_op_1 = builder.create<mlir::peng::GetTensorOp>(
        loc, tensor_type_1, buffer_op, 0);

    llvm::outs() << "get_tensor_op_1  " << get_tensor_op_1 << "\n";

    auto get_tensor_op_2 = builder.create<mlir::peng::GetTensorOp>(
        loc, tensor_type_2, buffer_op, 1);
    llvm::outs() << "get_tensor_op_2  " << get_tensor_op_2 << "\n";

    auto exp_op = builder.create<mlir::peng::ExpOp>(loc, get_tensor_op_2);
    llvm::outs() << "exp_op  " << exp_op << "\n";

    auto add_op = builder.create<mlir::peng::AddOp>(loc, get_tensor_op_1, get_tensor_op_2);
    llvm::outs() << "add_op  " << add_op << "\n";

    auto sub_op = builder.create<mlir::peng::SubOp>(loc, get_tensor_op_1, get_tensor_op_2);
    llvm::outs() << "sub_op  " << sub_op << "\n";

    auto out_buffer_op = builder.create<mlir::peng::BufferOp>(
        loc, mlir::ValueRange({const_2, const_4}));
    auto all_to_all_op = builder.create<mlir::peng::AllToAllOp>(
        loc, buffer_op, out_buffer_op);
    llvm::outs() << "all_to_all_op  " << all_to_all_op << "\n";

    auto print_op = builder.create<mlir::peng::PrintOp>(loc, const_1);
    llvm::outs() << "print_op  " << print_op << "\n";
}

mlir::ModuleOp getModule(mlir::OpBuilder& builder) {
    auto loc = builder.getUnknownLoc();
    auto context = builder.getContext();
    auto module = builder.create<mlir::ModuleOp>(loc, "Peng");
    builder.setInsertionPointToStart(module.getBody());
    auto f32 = mlir::Float32Type::get(context);
    auto dy_dim = 128;
    auto dy_shape = mlir::SmallVector<int64_t>({dy_dim, dy_dim, 24});
    auto dy_tensor_type =
        mlir::peng::PTensorType::get(context, dy_shape, f32, 0);
    auto func_type =
        mlir::FunctionType::get(context, {dy_tensor_type}, {dy_tensor_type});
    auto func =
        builder.create<mlir::func::FuncOp>(loc, KEntryPointName, func_type);
    func->setAttr(KHostFunc, builder.getUnitAttr());
    func->setAttr(KDPAttrName,
                  mlir::peng::DataParallelismAttr::get(context, 2));

    auto block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    // Softmax Op
    mlir::Value softmax_op = builder.create<mlir::peng::SoftmaxOp>(
        loc, block->getArgument(0), 1);
    softmax_op = builder.create<mlir::peng::SoftmaxOp>(loc, softmax_op, 1);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{softmax_op});
    return module;
}


int main() { CH3(); }
