//
// Created by peng on 9/12/25.
//
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Value.h"
#include <iostream>
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


    ::llvm::LogicalResult BufferOp::verify() {
        auto tensors = getTensors();
        auto devices = cast<BufferType>(getType()).getDevices();
        if (tensors.size() == 0) return llvm::failure();
        for (auto [index, device_id, tensor] : llvm::enumerate(devices, tensors)) {
            auto tensor_type = cast_or_null<PTensorType>(tensor.getType());
            if (device_id != tensor_type.getDeviceId()) {
                std::cout << "error " << std::endl;
                return llvm::failure();
            }
        }
        return llvm::success();
    }
}