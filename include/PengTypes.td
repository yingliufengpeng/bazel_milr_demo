
#ifndef DIALECT_PENG_TYPE
#define DIALECT_PENG_TYPE
include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "PengDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class Peng_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Peng_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName =  dialect.name # "." # typeMnemonic;
}


def Peng_TensorType : Peng_Type<"PTensor","p_tensor",[]>{
  // 概述
  let summary = " the summary of peng tensor type";

  // 方言的详细描述
  let description = "description of peng tensor type";

  // 参数
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "int64_t":$device_id
  );

  // 是否生成StorageClass, 无特殊情况，建议设为ture
  let genStorageClass = 1;

  // 不建议改动
  let hasStorageCustomConstructor = 0;

  // 额外的builder 声明
  let builders = [
    TypeBuilder<(ins
        "::mlir::ArrayRef<int64_t>":$shape,
        "::mlir::Type":$elementType),[{
      return $_get(elementType.getContext(), shape, elementType, 0);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
//   let assemblyFormat = "`<`$shape`,`$elementType`,`$device_id`>`";

  // 跳过默认的builder函数
  let skipDefaultBuilders = 0;

  // 是否生成类型检验的函数声明
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{

  }];
}




#endif // DIALECT_PENG_TYPE
