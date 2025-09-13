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
void CH2() {
    mlir::DialectRegistry DialectRegistry;
    mlir::MLIRContext context(DialectRegistry);
    auto diaglect = context.getOrLoadDialect<mlir::peng::PengDialect>();
    diaglect->sayHello();
    auto f32 = mlir::Float32Type::get(&context);
    llvm::outs() << f32;
    llvm::outs() << "\n";
    auto peng_tensor = mlir::peng::PTensorType::get(&context, {3, 4}, mlir::Float64Type::get(&context), 3);
    llvm::outs() << "peng_tensor" << "\t";
    peng_tensor.dump();
    llvm::outs() << "\n";
    mlir::peng::Layout m = mlir::peng::Layout::LEFT;
    llvm::outs() << "enum left is " << m << "\t" << "\n";

    // DataParallelismAttr
    auto dp_attr = mlir::peng::DataParallelismAttr::get(&context, 3);
    llvm::outs() << "DataParallelism Attribute :\t";
    dp_attr.dump();

    auto layout_atrr = mlir::peng::LayoutAttr::get(&context, m);

    llvm::outs() << "DataParallelism Attribute :\t";
    layout_atrr.dump();
}

void CH3() {
    mlir::DialectRegistry DialectRegistry;
    mlir::MLIRContext context(DialectRegistry);
    auto diaglect = context.getOrLoadDialect<mlir::peng::PengDialect>();

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();


    // ModuleOp
    auto module = builder.create<mlir::ModuleOp>(loc, "PengDialect");
    builder.setInsertionPointToStart(module.getBody());
    // ConstOp
    auto f32 = mlir::Float32Type::get(&context);
    auto shape = mlir::SmallVector<int64_t>({2, 2});
    auto const_value_1 =
            mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float) 1));
    auto const_value_2 =
            mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float) 2));
    auto tensor_type_1 =
            mlir::peng::PTensorType::get(&context, shape, f32, 0);
    auto tensor_type_2 =
            mlir::peng::PTensorType::get(&context, shape, f32, 1);
    auto const_1 = builder.create<mlir::peng::ConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_1));

    llvm::outs() << "const_1  " << const_1 << "\n";

    auto const_2 = builder.create<mlir::peng::ConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_1));

    llvm::outs() << "const_2  " << const_2 << "\n";


    auto const_3 = builder.create<mlir::peng::ConstOp>(
        loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_2));

    llvm::outs() << "const_3  " << const_3 << "\n";

    auto const_4 = builder.create<mlir::peng::ConstOp>(
        loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_2));

    llvm::outs() << "const_4  " << const_4 << "\n";

    // Buffer Op
    auto buffer_op = builder.create<mlir::peng::BufferOp>(
        loc, mlir::ValueRange({const_2, const_4,}));
    llvm::outs() << "buffer_op  " << buffer_op << "\n";


    // Get Tensor Op
    auto get_tensor_op_1 = builder.create<mlir::peng::GetTensorOp>(
        loc, tensor_type_1, buffer_op, 0);

    llvm::outs() << "get_tensor_op_1  " << get_tensor_op_1 << "\n";

    auto get_tensor_op_2 = builder.create<mlir::peng::GetTensorOp>(
        loc, tensor_type_2, buffer_op, 1);
    llvm::outs() << "get_tensor_op_2  " << get_tensor_op_2 << "\n";

    auto exp_op = builder.create<mlir::peng::ExpOp>(loc, get_tensor_op_2);
    llvm::outs() << "exp_op  " << exp_op << "\n";

    auto add_op = builder.create<mlir::peng::AddOp>(loc, get_tensor_op_1, get_tensor_op_2);
    llvm::outs() << "add_op  " << add_op << "\n";

    auto sub_op = builder.create<mlir::peng::SubOp>(loc, get_tensor_op_1, get_tensor_op_2);
    llvm::outs() << "sub_op  " << sub_op << "\n";

    auto out_buffer_op = builder.create<mlir::peng::BufferOp>(
        loc, mlir::ValueRange({const_2, const_4}));
    auto all_to_all_op = builder.create<mlir::peng::AllToAllOp>(
        loc, buffer_op, out_buffer_op);
    llvm::outs() << "all_to_all_op  " << all_to_all_op << "\n";

    auto print_op = builder.create<mlir::peng::PrintOp>(loc, const_1);
    llvm::outs() << "print_op  " << print_op << "\n";
}

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
    llvm::outs() << "NSTensorType: \t";
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


void CH8() {  // 初始化方言注册器
    mlir::DialectRegistry registry;
    // 初始化上下文环境
    mlir::MLIRContext context(registry);
    // 加载/注册方言
    context.getOrLoadDialect<mlir::peng::PengDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = getModule(builder);
    std::cout << "ops = " << module.getOps().empty() << std::endl;
    mlir::PassManager pm(&context);
    mlir::peng::MarkDistributeParallelParametersPassOptions
        mark_distribute_parallel_option{.DPNums = 3, .TPNums = 1};
    pm.addPass(mlir::peng::createMarkDistributeParallelParametersPass(
        mark_distribute_parallel_option));
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::peng::createApplyDistributeTransformPass());
    module->dump();
    if (pm.run(module).failed()) {
        llvm::outs() << "run pass error!\n";
    };
    llvm::outs() << "after pass:\n";
    module->dump();
}
void CH9() {
    mlir::DialectRegistry registry;
    // 初始化上下文环境
    mlir::MLIRContext context(registry);
    context.disableMultithreading(true);
    // 加载/注册方言
    context.getOrLoadDialect<mlir::peng::PengDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = getModule(builder);
    mlir::PassManager pm(&context);
    mlir::peng::MarkDistributeParallelParametersPassOptions
        mark_distribute_parallel_option{.DPNums = 3, .TPNums = 1};
    pm.addPass(mlir::peng::createMarkDistributeParallelParametersPass(
        mark_distribute_parallel_option));
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::peng::createApplyDistributeTransformPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::peng::createDeviceRegionFusionPass());
    module->dump();
    if (pm.run(module).failed()) {
        llvm::outs() << "run pass error!\n";
    };
    llvm::outs() << "after pass:\n";
    module->dump();
}            

int main() { CH9(); }
