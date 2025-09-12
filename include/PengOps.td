#ifndef DIALECT_PENG_OPS_TD
#define DIALECT_PENG_OPS_TD
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "PengDialect.td"
include "PengConstraints.td"

class Peng_Op<string mnemonic, list<Trait> traits = []>
    : Op<Peng_Dialect, mnemonic, traits> {
  let summary = cppNamespace#opName#"op";

  let description = "$_name op";
}


class Peng_UnaryOp<string mnemonic,Type OperandType, Type resultType = OperandType,list<Trait> traits = [], dag attributes = (ins)>:
    Peng_Op<mnemonic, traits#[]>{
    let arguments = !con((ins
        OperandType:$input),
        attributes
        );

    let results = (outs
        resultType:$result);
}


class Peng_BinaryOp<string mnemonic,Type OperandType, Type resultType = OperandType,list<Trait> traits = [], dag attributes = (ins)>:
    Peng_Op<mnemonic, traits#[]>{
    let arguments = !con((ins
        OperandType:$lhs,
        OperandType:$rhs),
        attributes
        );

    let results = (outs
        resultType:$result);
}

def Peng_ConstOp : Peng_Op<"const",[]>{
    let arguments = (ins
        ElementsAttr:$value);
    let results = (outs
        AnyNSTensor:$result);
}


#endif // DIALECT_PENG_OPS_TD