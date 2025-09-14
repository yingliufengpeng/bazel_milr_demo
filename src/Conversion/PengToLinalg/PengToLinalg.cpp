 

#include <memory>

#include "include/PengOps.h"
#include "include/PengTypes.h"
#include "include/PengDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
namespace {
struct SoftmaxOpToLinalgPattern final
    : public OpConversionPattern<mlir::peng::SoftmaxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(peng::SoftmaxOp op) const final {
    return llvm::success();
  }
  void rewrite(peng::SoftmaxOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto convert = getTypeConverter();
    llvm::SmallVector<Value> out_dy_sizes;
    auto input = adaptor.getInput();
    auto res_type =
        llvm::dyn_cast_or_null<ShapedType>(convert->convertType(op.getType()));
    auto rank = res_type.getRank();
    for (auto i : llvm::index_range(0, rank)) {
      if (!res_type.isDynamicDim(i)) continue;
      auto dim = rewriter.create<tensor::DimOp>(loc, input, i);
      out_dy_sizes.push_back(dim.getResult());
    }
    auto output = rewriter.create<tensor::EmptyOp>(
        loc, res_type.getShape(), res_type.getElementType(), out_dy_sizes);
    auto new_softmax = rewriter.create<linalg::SoftmaxOp>(
        loc, res_type, adaptor.getInput(), output, adaptor.getAxis());
    rewriter.replaceOp(op, new_softmax);
  }
};

struct DeviceKernelOpConvertPattern final
    : public OpConversionPattern<mlir::peng::DeviceKernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(peng::DeviceKernelOp op) const final {
    return llvm::success();
  }
  void rewrite(peng::DeviceKernelOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    llvm::SmallVector<Type> new_results;
    if (getTypeConverter()
            ->convertTypes(op.getResultTypes(), new_results)
            .failed()) {
      return;
    };
    auto new_op = rewriter.create<peng::DeviceKernelOp>(
        loc, new_results, adaptor.getSymName(), adaptor.getDeviceId(),
        adaptor.getArgs());
    rewriter.cloneRegionBefore(op.getRegion(), new_op.getRegion(),
                               new_op.getRegion().end());
    auto new_block = new_op.getBody();
    for (auto [index, arg] : llvm::enumerate(new_block->getArguments())) {
      if (auto ns_tensor =
              llvm::dyn_cast_or_null<peng::PTensorType>(arg.getType())) {
        rewriter.setInsertionPointAfterValue(arg);
        arg.setType(RankedTensorType::get(ns_tensor.getShape(),
                                          ns_tensor.getElementType()));
        auto cast = rewriter.create<UnrealizedConversionCastOp>(
            loc, ns_tensor, new_block->getArgument(index));
        rewriter.replaceAllUsesExcept(arg, cast.getResult(0), cast);
      }
    }
    auto return_op = new_block->getTerminator();
    for (auto [index, operand] : llvm::enumerate(return_op->getOperands())) {
      if (auto ns_tensor = llvm::dyn_cast_or_null<peng::PTensorType>(
              operand.getType())) {
        rewriter.setInsertionPointAfterValue(operand);
        auto cast = rewriter.create<UnrealizedConversionCastOp>(
            loc, typeConverter->convertType(operand.getType()), operand);
        return_op->setOperand(index, cast.getResult(0));
      }
    }
    for (auto [index, res, new_res] :
         llvm::enumerate(op->getResults(), new_op->getResults())) {
      rewriter.setInsertionPointAfterValue(new_res);
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          loc, res.getType(), new_res);
      rewriter.replaceAllUsesWith(res, cast.getResult(0));
    }
    rewriter.replaceOp(op, new_op);
  };
};
}  // namespace

namespace mlir::peng {
namespace {

static Value materializeToPTensor(OpBuilder &builder, PTensorType type,
                                   ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(isa<RankedTensorType>(inputs[0].getType()));
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(isa<PTensorType>(inputs[0].getType()));
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

}  // namespace
void initPengToLinalgTypeConvert(TypeConverter &typeConverter) {
  typeConverter.addConversion([](PTensorType type) {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });
  typeConverter.addSourceMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1) return std::nullopt;

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });
  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1) return std::nullopt;

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });
}
void populatePengToLinalgPatterns(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
                                
  patterns.add<SoftmaxOpToLinalgPattern, DeviceKernelOpConvertPattern>(
      typeConverter, patterns.getContext());
};
}  // namespace mlir::peng
