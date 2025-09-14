
#include <memory>

#include "include/Conversion/PengToLinalg/PengToLinalg.h"


#include "include/PengDialect.h"
#include "include/PengOps.h"
#include "include/PengTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "convert-north-satr-to-linalg"

namespace mlir::peng {

#define GEN_PASS_DEF_CONVERTPENGTOLINALGPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir::peng

using namespace ::mlir;
using namespace ::mlir::peng;

struct PengToLinalgPassPass
    : public mlir::peng::impl::ConvertPengToLinalgPassBase<
          PengToLinalgPassPass> {
  void runOnOperation() override;
};

void configPengToLinalgTarget(ConversionTarget& target) {
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<BufferCastOp>();
  target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
    for (auto type : op->getOperandTypes()) {
      if (isa<::mlir::peng::PTensorType>(type)) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<DeviceKernelOp>([](DeviceKernelOp op) {
    for (auto type : op.getArgs().getTypes()) {
      if (isa<::mlir::peng::PTensorType>(type)) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<SoftmaxOp>([](Operation* op) {
    return !llvm::isa<DeviceKernelOp>(op->getParentOp());
  });
}
void PengToLinalgPassPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  auto model = getOperation();
  TypeConverter type_convert;
  initPengToLinalgTypeConvert(type_convert);
  RewritePatternSet patterns(&getContext());
  populatePengToLinalgPatterns(type_convert, patterns);
  ConversionTarget target(getContext());
  configPengToLinalgTarget(target);
  if (failed(applyPartialConversion(model, target, std::move(patterns))))
    signalPassFailure();
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}