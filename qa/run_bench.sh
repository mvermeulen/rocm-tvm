#!/bin/bash
#
# Run tvm benchmarks to collect results
#
# Expected output:
#         --------------------------------------------------
#         Network Name         Mean Inference Time (std dev)
#         --------------------------------------------------
#         Cannot find config for target=rocm --model=1080ti,...
#         resnet-18            x.xx ms             (y.yy ms)
# 
# Variable defaults come from dockerfile container
TVM_HOME=${TVM_HOME:="/src/tvm"}
ROCM_TVM_HOME=${ROCM_TVM_HOME:="/src/rocm-tvm"}

cd ${ROCM_TVM_HOME}/apps
python3 gpu_imagenet_bench.py --target=rocm --network=resnet-18

