
#ifndef DIALECT_PENG_EUNMS_TD
#define DIALECT_PENG_EUNMS_TD

include "mlir/IR/EnumAttr.td"
include "PengTypes.td"


def Peng_LAYOUT_LEFT        : I32EnumAttrCase<"LEFT", 0>;
def Peng_LAYOUT_RIGHT         : I32EnumAttrCase<"RIGH", 1>;//C last
def Peng_LAYOUT_NCHW         : I32EnumAttrCase<"NCHW", 2>;
def Peng_LAYOUT_NHWC         : I32EnumAttrCase<"NHWC", 3>;//C last

def Peng_Layout : I32EnumAttr<"Layout",
    "Layout of tensor",
    [Peng_LAYOUT_LEFT, Peng_LAYOUT_RIGHT, Peng_LAYOUT_NCHW, Peng_LAYOUT_NHWC]> {

  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::peng";
}



#endif //DIALECT_PENG_EUNMS_TD