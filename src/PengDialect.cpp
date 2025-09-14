//
// Created by peng on 9/12/25.
//
#include "mlir/Dialect/Tensor/IR/Tensor.h"


#include "../include/PengDialect.h"
#include "../include/PengOps.h"

#include "PengDialect.cpp.inc"

void mlir::peng::PengDialect::initialize() {
    llvm::outs() << "initialize " << getDialectNamespace() << "  Type\n";
    registerTypes();
    registerAttrs();
    registerOps();

}


namespace mlir::peng {

    void PengDialect::sayHello() {
        llvm::outs() << "say hello" << " \n";
    }

    ::mlir::Operation *PengDialect::materializeConstant(
    ::mlir::OpBuilder &builder, ::mlir::Attribute value, ::mlir::Type type,
    ::mlir::Location loc) {
        llvm::outs() << __func__ << "\n";
        if (isa<::mlir::ElementsAttr>(value)) {
            return builder.create<mlir::peng::ConstOp>(loc, type,
                                           llvm::cast<::mlir::ElementsAttr>(value));
        }
        return nullptr;
    }
}