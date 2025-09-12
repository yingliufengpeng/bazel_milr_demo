
#ifndef INTERFACES_DISTRIBUTE_PARALLELISM_OP_INTERFACES_TD
#define INTERFACES_DISTRIBUTE_PARALLELISM_OP_INTERFACES_TD
include "mlir/IR/Interfaces.td"


def SupportedDataParallelismOp: OpInterface<"SupportedDataParallelismOp">{
  let description = "Properties related to data parallelism";
  let cppNamespace = "::mlir::peng";
  let methods = [
    InterfaceMethod<
      /*desc=*/        "进行数据并行的变换",
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "applyDataParallelism",
      /*args=*/        (ins "::mlir::peng::DistributeParallelAttr":$attr),
      /*methodBody=*/  "",
      /*defaultImpl=*/ [{
      return llvm::failure();
      }]
      >,
      InterfaceMethod<
      /*desc=*/        "进行数据并行的变换",
      /*returnType=*/  "bool",
      /*methodName=*/  "supportedDataParallelism",
      /*args=*/        (ins),
      /*methodBody=*/  "",
      /*defaultImpl=*/ [{
      Operation* op = $_op.getOperation();
      if (op->getNumOperands() == 0) return true;
      auto base_type = op->getOperand(0).getType();
      if (!isa<mlir::ShapedType>(base_type)) return false;
      for (auto operand : op->getOperands()) {
        if (operand.getType() != base_type) return false;
      }
      return true;
      }]
      >
      ];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

def DistributeParallelOp:OpInterface<"DistributeParallelOp">{
  let description = "Properties related to distribute parallelism";
  let cppNamespace = "::mlir::peng";
  let methods = [];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = [{
    // 实现并行变换
    ::mlir::LogicalResult applyDistributeParallelism(const ::mlir::peng::DistributeParallelAttr attr){
      if (isa<mlir::peng::DataParallelAttr>(attr)) {
        if (!isa<mlir::peng::SupportedDataParallelismOp>($_op.getOperation())) return ::llvm::failure();
        return dyn_cast<mlir::peng::SupportedDataParallelismOp>($_op.getOperation()).applyDataParallelism(attr);
      } else {
        llvm_unreachable("unsupported parallel type!");
      }
      return ::llvm::failure();
    };

    bool supportedDistributeParallelism(){
      if (isa<mlir::peng::SupportedDataParallelismOp>($_op.getOperation())){
        return dyn_cast<mlir::peng::SupportedDataParallelismOp>($_op.getOperation()).supportedDataParallelism();
      }else{
        llvm_unreachable("unsupported parallel type!");
      }
      return false;
    }
  }];
}

#endif // INTERFACES_DISTRIBUTE_PARALLELISM_OP_INTERFACES_TD