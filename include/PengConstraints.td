
#ifndef DIALECT_PENG_CONSTRAINTS_TD
#define DIALECT_PENG_CONSTRAINTS_TD
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def AnyPTensor : Type<And<[CPred<"::mlir::isa<::mlir::peng::PTensorType>($_self)">,
                            CPred<"::mlir::cast<::mlir::peng::PTensorType>($_self).getShape().size() > 0">]>>;

def AnyBuffer  : Type<And<[CPred<"::mlir::isa<::mlir::peng::BufferType>($_self)">,
                            CPred<"::mlir::cast<::mlir::peng::BufferType>($_self).getDevices().size() > 0">]>>;

//def NSTensorOrBuffer :AnyTypeOf<[AnyNSTensor,AnyBuffer]>;
#endif // DIALECT_PENG_CONSTRAINTS_TD 