
#include <memory>

#include "../../include/PengDialect.h"
#include "../../include/PengAttrs.h"
#include "../../include/Transforms/Passes.h"
#include "../../include/DistributeParallelismInterfaces.h"
#include "../../include/Utils/Key.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::peng {
#define GEN_PASS_DEF_MARKDISTRIBUTEPARALLELPARAMETERSPASS
#include "Transforms/Passes.h.inc"

}  // namespace mlir::peng
using namespace ::mlir;
using namespace ::mlir::peng;

struct MarkDistributeParallelParametersPass
    : ::mlir::peng::impl::MarkDistributeParallelParametersPassBase<
          MarkDistributeParallelParametersPass> {
  using MarkDistributeParallelParametersPassBase<
      MarkDistributeParallelParametersPass>::
      MarkDistributeParallelParametersPassBase;
  void runOnOperation() override;
};

void MarkDistributeParallelParametersPass::runOnOperation() {
  llvm::outs() << "run in: " << getPassName() << "\n";
  auto module = getOperation();
  llvm::outs() << "root op: " << module->getName() << "\n";
  llvm::outs() << "DPNums: " << DPNums << "\n";
  llvm::outs() << "TPNums: " << TPNums << "\n";
  llvm::outs() << "EPNums: " << EPNums << "\n";

  if (TPNums != 1) llvm::errs() << "TPNums not supported currently!\n";
  if (DPNums != 1) {
    auto dp_attr = DataParallelismAttr::get(&getContext(), DPNums);
    module->walk([&dp_attr](func::FuncOp op) {
      op->setAttr(KDPAttrName, dp_attr);
    });
  }
  llvm::outs() << "run out: " << getPassName() << "\n\n";
}
