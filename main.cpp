
#include <iostream>

#include "include/PengDialect.h"
#include "include/PengTypes.h"
#include "include/PengAttrs.h"
#include "include/PengEnums.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

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
    auto dp_attr = mlir::peng::DataParallelismAttr::get(&context, 2);
    llvm::outs() << "DataParallelism Attribute :\t";
    dp_attr.dump();

}

int main() { CH2(); }