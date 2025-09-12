
#ifndef DIALECT_PENG_EUNMS_TD
#define DIALECT_PENG_EUNMS_TD

include "mlir/IR/EnumAttr.td"
include "PengTypes.td"


def Peng_LEFT        : I32EnumAttrCase<"LEFT", 0>;
def Peng_RIGHT         : I32EnumAttrCase<"RIGH", 1>;//C last


def Peng_Layout : I32EnumAttr<"Layout",
    "Layout of tensor",
    [Peng_LEFT, Peng_RIGHT]> {

  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::peng";
}



#endif //DIALECT_PENG_EUNMS_TD