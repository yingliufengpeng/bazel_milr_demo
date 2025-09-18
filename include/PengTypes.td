
#ifndef DIALECT_PENG_TYPE
#define DIALECT_PENG_TYPE
include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "PengDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "PengTypeInterface.td"

class Peng_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Peng_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName =  dialect.name # "." # typeMnemonic;
}


def Peng_TensorType : Peng_Type<"PTensor","p_tensor",[ShapedTypeInterface],"::mlir::TensorType">{
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
  // let assemblyFormat = "`<`$shape`,`$elementType`,`$device_id`>`";

  // 跳过默认的builder函数
  let skipDefaultBuilders = 0;

  // 是否生成类型检验的函数声明
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    using ::mlir::ShapedType::Trait<PTensorType>::getElementTypeBitWidth;
    using ::mlir::ShapedType::Trait<PTensorType>::getRank;
    using ::mlir::ShapedType::Trait<PTensorType>::getNumElements;
    using ::mlir::ShapedType::Trait<PTensorType>::isDynamicDim;
    using ::mlir::ShapedType::Trait<PTensorType>::hasStaticShape;
    using ::mlir::ShapedType::Trait<PTensorType>::getNumDynamicDims;
    using ::mlir::ShapedType::Trait<PTensorType>::getDimSize;
    using ::mlir::ShapedType::Trait<PTensorType>::getDynamicDimIndex;

    ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type type){
        if(shape)
          return mlir::peng::PTensorType::get(getContext(), *shape, type);
        return mlir::peng::PTensorType::get(getContext(), getShape(), type);
    }
    PTensorType clone(::mlir::ArrayRef<int64_t> shape,int64_t device_id) const {
      return PTensorType::get(getContext(), shape, getElementType(),device_id);
    }
    PTensorType clone(::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType) const {
      return PTensorType::get(getContext(), shape, elementType,getDeviceId());
    }
    PTensorType clone(::mlir::ArrayRef<int64_t> shape) const {
      return PTensorType::get(getContext(), shape, getElementType(),getDeviceId());
    }
    PTensorType clone(::mlir::Type elementType) const {
      return PTensorType::get(getContext(), getShape(), elementType,getDeviceId());
    }
    PTensorType clone() const {
      return PTensorType::get(getContext(), getShape(), getElementType(),getDeviceId());
    }
    }];
}


def Peng_BufferType : Peng_Type<"Buffer","buffer",[]>{
  let summary = " the summary of north-star buffer type";

  let description = "description of north-star buffer type";

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$devices
  );

  let genStorageClass = 1;

  let hasStorageCustomConstructor = 0;

  let assemblyFormat = "`<`$devices`>`";

  let skipDefaultBuilders = 0;

  let genVerifyDecl = 1;
}



def MyIntRangeType : Peng_Type<"MyIntRange", "my_int_range", [HasSizeInterface]> {
//  let cppNamespace = "::mlir::peng";
     let parameters = (ins
        "int64_t":$low_index,
        "int64_t":$high_index
      );

  // 指定存储类
  let storageClass = "MyIntRangeTypeStorage";

  let extraClassDeclaration = [{
        int64_t getSize();
  }];

  let assemblyFormat = "`<`$low_index`x`$high_index`>`";

}

#endif // DIALECT_PENG_TYPE
