//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//


#include "include/PengDialect.h"
#include "include/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::peng::PengDialect>();
  registerAllExtensions(registry);
  mlir::peng::registerPengDialectOptPasses();
  // mlirEnableGlobalDebug(true);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Peng modular optimizer driver\n", registry));
}
