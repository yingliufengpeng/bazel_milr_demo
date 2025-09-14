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

void attributeBrief() {
    auto context = new mlir::MLIRContext;
    context->getOrLoadDialect<mlir::peng::PengDialect>();

    // Float Attr  表示浮点数的Attribute
    auto f32_attr = mlir::FloatAttr::get(mlir::Float32Type::get(context), 2);
    llvm::outs() << "F32 Attribute :\t";
    f32_attr.dump();

    // Integer Attr  表示整数的Attribute
    auto i32_attr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), 10);
    llvm::outs() << "I32 Attribute :\t";
    i32_attr.dump();

    // StrideLayout Attr  表示内存布局信息的Attribute
    auto stride_layout_attr = mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1});
    llvm::outs() << "StrideLayout Attribute :\t";
    stride_layout_attr.dump();

    // String Attr    表示字符串的Attribute
    auto str_attr = mlir::StringAttr::get(context, "Hello, MLIR!");
    llvm::outs() << "String Attribute :\t";
    str_attr.dump();

    // StrRef Attr   表示符号的Attribute
    auto str_ref_attr = mlir::SymbolRefAttr::get(str_attr);
    llvm::outs() << "SymbolRef Attribute :\t";
    str_ref_attr.dump();

    // Type Attr    储存Type 的Attribute
    auto type_attr = mlir::TypeAttr::get(mlir::peng::PTensorType::get(
        context, {1, 2, 3}, mlir::Float32Type::get(context)));
    llvm::outs() << "Type Attribute :\t";
    type_attr.dump();

    // Unit Attr   一般作为标记使用
    auto unit_attr = mlir::UnitAttr::get(context);
    llvm::outs() << "Unit Attribute :\t";
    unit_attr.dump();

    auto i64_arr_attr = mlir::DenseI64ArrayAttr::get(context, {1, 2, 3});
    llvm::outs() << "Array Attribute :\t";
    i64_arr_attr.dump();

    auto dense_attr = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get({2, 2}, mlir::Float32Type::get(context)),
        llvm::ArrayRef<float>{1, 2, 3, 4});
    llvm::outs() << "Dense Attribute :\t";
    dense_attr.dump();
    delete context;
}


void CH4() {
  attributeBrief();
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  auto dialect = context.getOrLoadDialect<mlir::peng::PengDialect>();
  // Layout Eunms
  auto nchw = mlir::peng::Layout::NCHW;
  llvm::outs() << "NCHW: " << mlir::peng::stringifyEnum(nchw) << "\n";
  // LayoutAttr
  auto nchw_attr = mlir::peng::LayoutAttr::get(&context, nchw);
  llvm::outs() << "NCHW LayoutAttribute :\t";
  nchw_attr.dump();
  // DataParallelismAttr
  auto dp_attr = mlir::peng::DataParallelismAttr::get(&context, 2);
  llvm::outs() << "DataParallelism Attribute :\t";
  dp_attr.dump();
}



int main() { CH4(); }
