

#ifndef CONVERSION_PengTOLINALG_PengTOLINALG_H
#define CONVERSION_PengTOLINALG_PengTOLINALG_H
#include <memory>

#include "mlir/Pass/Pass.h"
namespace mlir {
class TypeConverter;
}

namespace mlir::peng {

void initPengToLinalgTypeConvert(TypeConverter &typeConverter);

void populatePengToLinalgPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);

// #define GEN_PASS_DECL_CONVERTPENGTOLINALGPASS
// #include "Conversion/Passes.h.inc"

}  // namespace mlir::peng
#endif  // CONVERSION_PengTOLINALG_PengTOLINALG_H