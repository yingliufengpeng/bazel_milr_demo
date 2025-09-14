
#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD
include "mlir/Pass/PassBase.td"

def ConvertPengToLinalgPass : Pass<"convert-peng-to-linalg","::mlir::ModuleOp"> {
  let summary = "标记全局并行参数";
  let description = [{
    "标记全局并行参数。";
  }];
  let dependentDialects = [
    "::mlir::peng::PengDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect"
  ];
}


#endif  // CONVERSION_PASSES_TD