//
// Created by peng on 9/14/25.
//

#ifndef BAZEL_MLIR_DEMO_CONVERSION_PASSES_H
#define BAZEL_MLIR_DEMO_CONVERSION_PASSES_H
#include "mlir/Pass/Pass.h"
namespace mlir::peng {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
}  //
#endif //BAZEL_MLIR_DEMO_CONVERSION_PASSES_H