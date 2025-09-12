
#include <iostream>
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "include/PengDialect.h"
#include "include/PengTypes.h"
#include "include/PengAttrs.h"
#include "include/PengEnums.h"
#include "include/PengAttrs.h"
#include "include/PengOps.h"


void CH2() {
    mlir::DialectRegistry DialectRegistry;
    mlir::MLIRContext context(DialectRegistry);
    auto diaglect = context.getOrLoadDialect<mlir::peng::PengDialect>();
    diaglect->sayHello();
    auto f32 = mlir::Float32Type::get(&context);
    llvm::outs() << f32;
    llvm::outs() << "\n";
    auto peng_tensor = mlir::peng::PTensorType::get(&context, {3, 4}, mlir::Float64Type::get(&context), 3);
    llvm::outs() << "peng_tensor" << "\t";
    peng_tensor.dump();
    llvm::outs() << "\n";
    mlir::peng::Layout m = mlir::peng::Layout::LEFT;
    llvm::outs() <<  "enum left is " << m << "\t" << "\n";

    // DataParallelismAttr
    auto dp_attr = mlir::peng::DataParallelismAttr::get(&context, 3);
    llvm::outs() << "DataParallelism Attribute :\t";
    dp_attr.dump();

    auto layout_atrr = mlir::peng::LayoutAttr::get(&context, m);

    llvm::outs() << "DataParallelism Attribute :\t";
    layout_atrr.dump();
}

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
        loc, mlir::ValueRange({const_2, const_4, }));
    llvm::outs() << "buffer_op  " << buffer_op << "\n";


}


int main() { CH3(); }