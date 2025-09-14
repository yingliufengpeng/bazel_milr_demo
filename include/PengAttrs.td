
#ifndef DIALECT_PENG_ATTRS_TD
#define DIALECT_PENG_ATTRS_TD

include "mlir/IR/EnumAttr.td"
include "PengEnums.td"
include "PengDialect.td"
include "DistributeParallelismAttrInterfaces.td"


class Peng_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<Peng_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = dialect.name # "." # attrMnemonic;
  let genStorageClass = 1;
  let hasStorageCustomConstructor = 0;
  let skipDefaultBuilders = 0;
  let genVerifyDecl = 0;
}

def Peng_DataParallelism: Peng_Attr<"DataParallelism", "DP", [DataParallelAttr]>{
  let parameters = (ins "int64_t":$DP_nums, ArrayRefParameter<"int64_t">:$devices);
  let builders = [
    AttrBuilder<(ins "int64_t":$DP_nums),
    [{
      llvm::SmallVector<int64_t> device_ids;
      for (auto i : llvm::index_range(0, DP_nums)) {
        device_ids.push_back(i);
      }
      return $_get($_ctxt, DP_nums, device_ids);
    }]
    >
  ];
  let assemblyFormat = [{
    `<`
      `DP` `=` $DP_nums `:` $devices
    `>`
  }];
}

def Peng_LayoutAttr : EnumAttr<Peng_Dialect, Peng_Layout, "LayoutAttr"> {
    let assemblyFormat = "`<` $value `>`";

    let extraClassDeclaration = [{
    bool isChannelLast();
    }];
}

#endif //DIALECT_PENG_ATTRS_TD