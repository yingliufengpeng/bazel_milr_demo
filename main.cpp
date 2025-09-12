
#include "include/PengDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

void CH2() {
    // 初始化方言注册器
    mlir::DialectRegistry registry;
    // 初始化上下文环境
    mlir::MLIRContext context(registry);
    // 加载/注册方言
    auto dialect = context.getOrLoadDialect<mlir::peng::PengDialect>();
    dialect->sayHello();
}

int main() { CH2(); }