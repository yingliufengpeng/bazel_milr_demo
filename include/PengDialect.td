

#ifndef DIALECT_PENG
#define DIALECT_PENG
include "mlir/IR/DialectBase.td"

def Peng_Dialect : Dialect {

    let name = "Peng";

    let summary = "";

    let description = "";

    let cppNamespace = "::mlir::peng";

    let extraClassDeclaration = [{
      void sayHello();
      void registerTypes();
      void registerAttrs();
      void registerOps();
    }];

    // 使用MLIR默认的类型解析输出.
    let useDefaultTypePrinterParser  = 1;

//
//    // 使用MLIR默认的属性解析输出.
  let useDefaultAttributePrinterParser = 1;

}

#endif // DIALECT_PENG
