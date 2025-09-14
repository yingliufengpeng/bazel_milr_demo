#include <iostream>
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "include/PengDialect.h"
#include "include/PengTypes.h"
#include "include/PengAttrs.h"
#include "include/PengEnums.h"
#include "include/PengAttrs.h"
#include "include/Transforms/Passes.h"
#include "include/PengOps.h"
#include "include/DistributeParallelismInterfaces.h"
#include "include/Utils/File.h"
#include "include/Utils/Key.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
namespace {

} // namespace




void IR_Struct() {
    const char* ir =
        R"(func.func @insertion_point_outside_loop(%t : tensor<?xf32>, %sz : index,
                                        %idx : index) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %blank = tensor.empty() : tensor<5xf32>

  %r = scf.for %iv = %c0 to %sz step %c5 iter_args(%bb = %t) -> (tensor<?xf32>) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %f = arith.sitofp %iv_i32 : i32 to f32

    %filled = linalg.fill ins(%f : f32) outs(%blank : tensor<5xf32>) -> tensor<5xf32>

    %inserted = tensor.insert_slice %filled into %bb[%idx][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %inserted : tensor<?xf32>
  }
  return %r : tensor<?xf32>
})";
    auto context = mlir::MLIRContext();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::affine::AffineDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (mlir::utils::file::ParseStr<mlir::ModuleOp>(context, module, ir).failed())
        llvm::outs() << " parse ir string failed!\n";
    auto file = std::filesystem::current_path() / "ir_struct.mlir";
    if (mlir::utils::file::dumpToFile(module.get(), file).failed()) {
        llvm::outs() << "print module error!";
    }
}

void CH7() { IR_Struct(); }




int main() { CH7(); }
