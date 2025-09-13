
#include "../../include/PengDialect.h"
#include "../../include/Transforms/Passes.h"
#include "../../include/DistributeParallelismInterfaces.h"
#include "../../include/Utils/Key.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::peng {

#define GEN_PASS_DEF_APPLYDISTRIBUTETRANSFORMPASS
#include "Transforms/Passes.h.inc"

}  // namespace mlir::peng
using namespace ::mlir;
using namespace ::mlir::peng;

struct ApplyDistributeTransformPass
    : ::mlir::peng::impl::ApplyDistributeTransformPassBase<
          ApplyDistributeTransformPass> {
  using ApplyDistributeTransformPassBase<
      ApplyDistributeTransformPass>::ApplyDistributeTransformPassBase;
  void runOnOperation() override;
};

void ApplyDistributeTransformPass::runOnOperation() {
  llvm::outs() << "run in: " << getPassName() << "\n";
  auto func = getOperation();
  llvm::outs() << "root op: " << func->getName() << "\n";
  auto dp_attr = llvm::dyn_cast_or_null<mlir::peng::DistributeParallelAttr>(
      func->getAttr(KDPAttrName));
  if (!dp_attr) llvm_unreachable("error!");
  func->walk([&](mlir::Operation* op) {
    if (auto dis_op = llvm::dyn_cast_or_null<mlir::peng::DistributeParallelOp>(op)) {
      if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
        llvm::outs() << "Apply DataParallelism to " << op->getName() << "\n";
        op->erase();
      };
    }
  });
  llvm::outs() << "run out: " << getPassName() << "\n\n";
}

std::unique_ptr<::mlir::Pass>
mlir::peng::createApplyDistributeTransformPass() {
  return std::make_unique<ApplyDistributeTransformPass>();
}