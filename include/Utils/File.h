
#ifndef UTILS_MLIR_UTILS_H
#define UTILS_MLIR_UTILS_H
#include <filesystem>
#include <system_error>

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
namespace mlir::utils::file {



template <class OpTy = Operation*>
inline llvm::LogicalResult PrintToFile(OpTy op, const char* file) {
  std::error_code error_code;
  auto file_dir = std::filesystem::path(file).parent_path();
  if (!std::filesystem::exists(file_dir)) {
    if (!std::filesystem::create_directory(file_dir)) {
      llvm::outs() << "create directory error!";
      return llvm::failure();
    }
  }
  llvm::raw_fd_stream file_stream(file, error_code);
  op->print(file_stream);
  llvm::outs() << "print " << op->getName() << " to " << file << "\n";
  return success();
}

template <class OpTy = Operation*>
mlir::LogicalResult dumpToFile(OpTy op, const std::filesystem::path &path) {
  std::string utf8Path = path.string();  // 在 Windows 下会做编码转换
  return PrintToFile(op, utf8Path.c_str());
}

template <class OpTy = Operation*>
inline llvm::LogicalResult ParseFile(mlir::MLIRContext& context,
                                     mlir::OwningOpRef<OpTy>& module,
                                     const char* file) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(file);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::outs() << "could not open input file: " << file;
    return failure();
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<OpTy>(sourceMgr, &context);
  if (!module) {
    llvm::outs() << "parse file error: " << file;
    return failure();
  }
  return success();
}

template <class OpTy = Operation*>
inline llvm::LogicalResult ParseStr(mlir::MLIRContext& context,
                                    mlir::OwningOpRef<mlir::ModuleOp>& module,
                                    const char* str) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getMemBuffer(str, "mlir_module");
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::outs() << "load ir string error!\n";
    return llvm::failure();
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<OpTy>(sourceMgr, {&context, false});
  if (!module) {
    llvm::outs() << "parse ir string fatal error!";
    return llvm::failure();
  }
  return success();
}

}  // namespace mlir::utils::file

#endif  // UTILS_MLIR_UTILS_H