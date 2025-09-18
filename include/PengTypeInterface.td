#ifndef INTERFACES_PEGN_TYPE_INTERFACES_TD
#define INTERFACES_PEGN_TYPE_INTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/DialectBase.td"



def HasSizeInterface : TypeInterface<"HasSizeInterface"> {
  let cppNamespace = "::mlir::peng";

  // 正确的方法列表写法
  let methods = [

    InterfaceMethod<[{
      getSize.
    }],
    "int64_t", "getSize">,
  ];
}


#endif //INTERFACES_PEGN_TYPE_INTERFACES_TD


