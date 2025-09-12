
#ifndef DIALECT_PENG_EUNMS_TD
#define DIALECT_PENG_EUNMS_TD

include "mlir/IR/EnumAttr.td"
include "PengTypes.td"


def Peng_LAYOUT_NCHW         : I32EnumAttrCase<"LEFT", 0>;
def Peng_LAYOUT_NHWC         : I32EnumAttrCase<"RIGH", 1>;//C last
def Peng_Layout : I32EnumAttr<"Layout",
    "Layout of tensor",
    [Peng_LAYOUT_NCHW, Peng_LAYOUT_NHWC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::peng";
}

//def Peng_LayoutAttr : EnumAttr<Peng_Dialect, Peng_Layout, "Layout2">{
//  let assemblyFormat = "`<` $value `>`";
//
//  let extraClassDeclaration = [{
//  bool isLeft();
//  }];
//}

#endif //DIALECT_PENG_EUNMS_TD