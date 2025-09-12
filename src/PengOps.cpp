//
// Created by peng on 9/12/25.
//
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Value.h"

#include "../include/PengDialect.h"
#include "../include/PengOps.h"

#define GET_OP_CLASSES
#include "PengOps.cpp.inc"

namespace mlir::peng {

    void PengDialect::registerOps() {
        llvm::outs() << "register " << getDialectNamespace() << "  Op\n";
        addOperations<
      #define GET_OP_LIST
      #include "PengOps.cpp.inc"
            >();
    }
}