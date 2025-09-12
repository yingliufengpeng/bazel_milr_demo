
#ifndef INTERFACES_DISTRIBUTE_PARALLELISM_ATTR_INTERFACES_TD
#define INTERFACES_DISTRIBUTE_PARALLELISM_ATTR_INTERFACES_TD
include "mlir/IR/Interfaces.td"

def DistributeParallelAttr: AttrInterface<"DistributeParallelAttr">{
  let description = "Properties related to distribute parallelism";
  let cppNamespace = "::mlir::peng";
  let methods = [];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

def DataParallelAttr: AttrInterface<"DataParallelAttr",[DistributeParallelAttr]>{
  let description = "Properties related to distribute parallelism";
  let cppNamespace = "::mlir::peng";
  let methods = [
    InterfaceMethod<[{
      DP 数量.
    }],
    "int64_t", "getDPNums">,

    InterfaceMethod<[{
      设备编号.
    }],
    "::llvm::ArrayRef<int64_t>", "getDevices">,
  ];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

#endif // INTERFACES_DISTRIBUTE_PARALLELISM_ATTR_INTERFACES_TD