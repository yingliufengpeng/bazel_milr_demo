
#ifndef DIALECT_PENG_ATTRS_TD
#define DIALECT_PENG_ATTRS_TD

include "mlir/IR/EnumAttr.td"
include "PengDialect.td"


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

def Peng_DataParallelism: Peng_Attr<"DataParallelism", "DP", []>{
  let parameters = (ins "int64_t":$DP_nums);
  let assemblyFormat = [{
    `<`
      `DP` `=` $DP_nums
    `>`
  }];
}

#endif //DIALECT_PENG_ATTRS_TD