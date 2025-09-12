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


    ::llvm::LogicalResult GetTensorOp::verify() {
        auto device_id = getDeviceId();
        auto buffer = getBuffer();
        if (isa<BlockArgument>(buffer)) {
            auto buffer_type = cast<BufferType>(buffer.getType());
            auto device_ids = buffer_type.getDevices();
            for (auto id : device_ids) {
                if (id == device_id) return llvm::success();
            }
            return llvm::failure();
        }
        auto buffer_op = llvm::cast_or_null<BufferOp>(buffer.getDefiningOp());
        if (!buffer_op) return llvm::failure();
        for (auto tensor : buffer_op.getTensors()) {
            auto tensor_type = cast_or_null<PTensorType>(tensor.getType());
            if (!tensor_type) return llvm::failure();
            if (device_id == tensor_type.getDeviceId()) {
                if (tensor_type != getType()) return llvm::failure();
                return llvm::success();
            }
        }
        return llvm::failure();
    };

    ::llvm::LogicalResult SoftmaxOp::verify() {
        auto axis = getAxis();
        if (axis < 0) return llvm::failure();
        auto input_type = cast<PTensorType>(getInput().getType());
        if (axis >= input_type.getShape().size()) return llvm::failure();
        return llvm::success();
    }
}