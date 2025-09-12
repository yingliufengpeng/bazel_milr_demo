//
// Created by peng on 9/12/25.
//
#include "mlir/Dialect/Tensor/IR/Tensor.h"


#include "../include/PengDialect.h"

#include "PengDialect.cpp.inc"

void mlir::peng::PengDialect::initialize() {
    llvm::outs() << "initialize " << getDialectNamespace() << "  Type\n";
    registerTypes();
    registerAttrs();

}


namespace mlir::peng {

    void PengDialect::sayHello() {
        llvm::outs() << "say hello" << " \n";
    }
}