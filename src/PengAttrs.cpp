//
// Created by peng on 9/12/25.
//
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "include/PengAttrs.h"
#include "mlir/IR/Attributes.h"
#include "include/PengDialect.h"

#define GET_ATTRDEF_CLASSES
#include "PengAttrs.cpp.inc"


namespace mlir::peng {

    void PengDialect::registerAttrs() {
        llvm::outs() << "register " << getDialectNamespace() << "  Attr\n";
        addAttributes<
          #define GET_ATTRDEF_LIST
          #include "PengAttrs.cpp.inc"
            >();
    }

    bool LayoutAttr::isLeft() {
        return true;
    }
}
