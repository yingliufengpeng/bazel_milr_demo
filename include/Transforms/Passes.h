//
// Created by peng on 9/13/25.
//

#ifndef BAZEL_MLIR_DEMO_PASSES_H
#define BAZEL_MLIR_DEMO_PASSES_H
#include "mlir/Pass/Pass.h"

namespace mlir::peng {
    std::unique_ptr<::mlir::Pass> createApplyDistributeTransformPass();



#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

}

#endif //BAZEL_MLIR_DEMO_PASSES_H