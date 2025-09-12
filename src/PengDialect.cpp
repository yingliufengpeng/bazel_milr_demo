//
// Created by peng on 9/12/25.
//
#include "mlir/Dialect/Tensor/IR/Tensor.h"


#include "../include/PengDialect.h"

#include "PengDialect.cpp.inc"

void mlir::peng::PengDialect::initialize() {


}
// 实现方言的析构函数
mlir::peng::PengDialect::~PengDialect() {
    llvm::outs() << "destroying " << getDialectNamespace() << "\n";
}

namespace mlir::peng {

    void PengDialect::sayHello() {
        llvm::outs() << "say hello" << " \n";
    }
}