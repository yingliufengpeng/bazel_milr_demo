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

mlir::ModuleOp getModule(mlir::OpBuilder& builder) {
    auto loc = builder.getUnknownLoc();
    auto context = builder.getContext();
    auto module = builder.create<mlir::ModuleOp>(loc, "Peng");
    builder.setInsertionPointToStart(module.getBody());
    auto f32 = mlir::Float32Type::get(context);
    auto dy_dim = 128;
    auto dy_shape = mlir::SmallVector<int64_t>({dy_dim, dy_dim, 24});
    auto dy_tensor_type =
        mlir::peng::PTensorType::get(context, dy_shape, f32, 0);
    auto func_type =
        mlir::FunctionType::get(context, {dy_tensor_type}, {dy_tensor_type});
    auto func =
        builder.create<mlir::func::FuncOp>(loc, KEntryPointName, func_type);
    func->setAttr(KHostFunc, builder.getUnitAttr());
    func->setAttr(KDPAttrName,
                  mlir::peng::DataParallelismAttr::get(context, 2));

    auto block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    // Softmax Op
    mlir::Value softmax_op = builder.create<mlir::peng::SoftmaxOp>(
        loc, block->getArgument(0), 1);
    softmax_op = builder.create<mlir::peng::SoftmaxOp>(loc, softmax_op, 1);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{softmax_op});
    return module;
}



void CH6() {
    // 初始化方言注册器
    mlir::DialectRegistry registry;
    // 初始化上下文环境
    mlir::MLIRContext context(registry);
    // 加载/注册方言
    context.getOrLoadDialect<mlir::peng::PengDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    // shaped type interface
    auto f32 = mlir::Float32Type::get(&context);
    auto dim = mlir::ShapedType::kDynamic;
    auto shape = mlir::SmallVector<int64_t>({dim, dim, 24});
    auto tensor_type =
            mlir::peng::PTensorType::get(&context, shape, f32, 0);
    auto shaped_type = mlir::cast<mlir::ShapedType>(tensor_type);
    llvm::outs() << "PTensorType: \t";
    tensor_type.dump();
    llvm::outs() << "Shaped Type Interface:\t";
    shaped_type.dump();
    auto cloned_type = shaped_type.clone(f32);
    llvm::outs() << "Cloned Shaped Type Interface:\t";
    cloned_type.dump();
    // Attr interface
    auto dp_attr = mlir::peng::DataParallelismAttr::get(&context, 2);
    llvm::outs() << dp_attr.getAbstractAttribute().getName()
            << " has mlir::DataParallelAttr interface: "
            << dp_attr.getAbstractAttribute().hasInterface(
                mlir::peng::DistributeParallelAttr::getInterfaceID())
            << "\n";
    llvm::outs()
            << dp_attr.getAbstractAttribute().getName()
            << " has mlir::DataParallelAttr interface: "
            << dp_attr.hasPromiseOrImplementsInterface<mlir::peng::DataParallelAttr>()
            << "\n";

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = getModule(builder);
    module->dump();
    module->walk([](mlir::func::FuncOp func) {
        std::cout << "aaaa" << std::endl;
        if (auto dp_attr = llvm::dyn_cast_or_null<mlir::peng::DistributeParallelAttr>(
            func->getAttr(KDPAttrName))) {
            std::cout << "bbbb" << std::endl;
            func->walk([&](mlir::Operation *op) {
                if (auto dis_op =
                        llvm::dyn_cast_or_null<mlir::peng::DistributeParallelOp>(op)) {
                    dis_op.supportedDistributeParallelism(); // for test
                    if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
                        llvm::outs() << "Apply DataParallelism to " << op->getName()
                                << "\n";
                        op->erase();
                    };
                }
            });
        }
    });
    module->dump();
}



int main() { CH6(); }
