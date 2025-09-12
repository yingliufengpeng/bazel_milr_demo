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

//     let hasVerifier = 1;
     let builders = [
        OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs),
            [{
                $_state.addOperands(lhs);
                $_state.addOperands(rhs);
                $_state.addTypes(lhs.getType());
            }]>

    ];

//      let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
//      let hasFolder = 1;
//      let hasCanonicalizer = 1;
}

def Peng_ConstOp : Peng_Op<"const",[]>{
    let arguments = (ins
        ElementsAttr:$value);
    let results = (outs
        AnyPTensor:$result);
}

def Peng_ExpOp : Peng_UnaryOp<"exp",AnyPTensor>{
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input) ,
            [{
                $_state.addOperands(input);
                $_state.addTypes(input.getType());
            }]>
    ];
}


def Peng_AddOp : Peng_BinaryOp<"add",AnyPTensor>;
def Peng_SubOp : Peng_BinaryOp<"sub",AnyPTensor>;
def Peng_MulOp : Peng_BinaryOp<"mul",AnyPTensor>;
def Peng_DivOp : Peng_BinaryOp<"div",AnyPTensor>;


def Peng_BufferOp : Peng_Op<"buffer",[]>{
    let description = "将 多个device_id 的tensor 组合成 一个 buffer";
    let arguments = (ins
        Variadic<AnyPTensor>:$tensors
    );
    let results = (outs
        AnyBuffer:$result);
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::ValueRange":$tensors) ,
            [{
                $_state.addOperands(tensors);
                ::llvm::SmallVector<int64_t> devices;
                for (auto tensor : tensors) {
                    auto tensor_type =
                        llvm::cast<::mlir::peng::PTensorType>(tensor.getType());
                    devices.push_back(tensor_type.getDeviceId());
                }
                $_state.addTypes(::mlir::peng::BufferType::get($_state.getContext(), devices));
            }]>
    ];
}


def Peng_GetTensorOp: Peng_Op<"get_tensor",[]>{
    let description = "从buffer中取出指定device_id的tensor";
    let arguments = (ins
        AnyBuffer:$buffer,
        I64Attr:$device_id
    );
    let results = (outs
        AnyPTensor:$result);
    let hasVerifier = 1;
}

def Peng_SoftmaxOp : Peng_UnaryOp<"softmax",AnyPTensor, AnyPTensor, [], (ins I64Attr:$axis)>{
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input, "int64_t":$axis) ,
            [{
                $_state.addOperands(input);
                $_state.getOrAddProperties<Properties>().axis = $_builder.getIntegerAttr(odsBuilder.getIntegerType(64,true), axis);
                $_state.addTypes(input.getType());
            }]>

    ];
}


def Peng_AllToAllOp : Peng_Op<"all_to_all",[]>{
    let arguments = (ins
        AnyBuffer:$input,
        AnyBuffer:$output
    );
}

def Peng_PrintOp: Peng_Op<"print",[]>{
    let arguments = (ins
        AnyType:$input
    );
}

#endif // DIALECT_PENG_OPS_TD