//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//


#include "include/PengDialect.h"
#include "include/Transforms/Passes.h"
#include "include/Conversion/Passes.h"
#include "include/Pipelines/Pipelines.h"
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
#include "mlir-c/Debug.h"
#include <iostream>
#include <filesystem>
#include <tuple>


#if defined(_MSC_VER)
#include "tools/cpp/runfiles/runfiles.h"
using rules_cc::cc::runfiles::Runfiles;

#endif

#include "mlir/Tools/mlir-opt/MlirOptMain.h"


std::string process_argv(char **argv) {

#if defined(_MSC_VER)

    auto arg0 = argv[0];
    auto file_name = argv[1];
    std::cout << "输入文件 :" << file_name << std::endl;
    std::unique_ptr<Runfiles> runfiles(Runfiles::Create(arg0));
    const std::string workspace_prefix = "_main/";

    std::string real_path = runfiles->Rlocation(workspace_prefix + file_name);


    return real_path;

#endif

    return argv[1];
}



int main(int argc, char **argv) {

    std::cout << "当前工作路径: " << std::filesystem::current_path() << std::endl;
    std::cout << "argv is  " << argv << std::endl;

    if (argc < 2) {
        std::cout << "参数传递错误  need file_name" << std::endl;
        return -1;
    }
     auto file_name = process_argv(argv);


    char* new_argv[2] = {
            argv[0],
            (char*)file_name.c_str(),


    };

    std::cout << "argv[0]  = " << argv[0] << std::endl;
    std::cout << "argv[1]  = " << argv[1] << std::endl;
    std::cout << "new_argv[1]  = " << new_argv[1] << std::endl;
    std::cout << "new_argv[1]  = " << new_argv[1] << std::endl;
    // assert(argv[0] == new_argv[0]);
    // assert(argv[1] == new_argv[1]);
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registerAllDialects(registry);
    registry.insert<mlir::peng::PengDialect>();
    registerAllExtensions(registry);
    mlir::peng::registerPengDialectOptPasses();
    mlir::peng::registerPengDialectConversionPasses();
    mlir::pipeline::registerPengBasicPipelines();
    mlirEnableGlobalDebug(true);

#if defined(_MSC_VER)

    auto m = mlir::MlirOptMain(argc, new_argv, "Peng modular optimizer driver\n", registry);

#else
    auto m = mlir::MlirOptMain(argc, argv, "Peng modular optimizer driver\n", registry);

#endif
    return mlir::asMainReturnCode(m);
}
