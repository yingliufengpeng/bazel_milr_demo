
#ifndef PIPELINES_PIPELINS_H
#define PIPELINES_PIPELINS_H
#include <cstdint>

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
namespace mlir::pipeline {

/// Options for the buffer deallocation pipeline.
struct PengBasicPipelineOptions
    : public PassPipelineOptions<PengBasicPipelineOptions> {
  PassOptions::Option<int64_t> DP_Nums{
      *this, "DP_Nums", llvm::cl::desc("数据并行参数."), llvm::cl::init(1)};
};

void buildPengBasicPipeline(
    OpPassManager &pm, const PengBasicPipelineOptions &options);

void registerPengBasicPipelines();

}  // namespace mlir::pipeline

#endif  // PIPELINES_PIPELINS_H