 
#include "include/Conversion/Passes.h"
#include "include/Transforms/Passes.h"
#include "include/Pipelines/Pipelines.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pipeline {
void buildBuffePengBasicPipeline(
    OpPassManager &pm, const PengBasicPipelineOptions &options) {
  mlir::peng::MarkDistributeParallelParametersPassOptions
      mark_distribute_parallel_option{.DPNums = options.DP_Nums, .TPNums = 1};
  pm.addPass(mlir::peng::createMarkDistributeParallelParametersPass(
      mark_distribute_parallel_option));
  pm.addNestedPass<func::FuncOp>(
      mlir::peng::createApplyDistributeTransformPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::peng::createDeviceRegionFusionPass());
  pm.addPass(mlir::peng::createConvertPengToLinalgPass());
};

void registerPengBasicPipelines() {
  PassPipelineRegistration<PengBasicPipelineOptions>(
      "north-star-basic-pipeline", "basic pipeline ",
      buildBuffePengBasicPipeline);
};

}  // namespace mlir::pipeline
