#!/bin/bash
#
# Run TVM quickstart example to make sure it works
#
# Expected output:
#    <dump of the neural network graph>
#    [0.00098592 0.00112624 0.00113839 0.00093255 0.00097393 0.00090567
#     0.00105719 0.0010534  0.00083118 0.00104978]
#    ['deploy_param.params', 'deploy_graph.json', 'deploy_lib.tar']
#    [0.0009989  0.00108    0.00106431 0.00094831 0.00100184 0.00091658
#     0.00103067 0.00102476 0.00091256 0.00105041]
#
# Exact values might differ slightly, but we should get a numeric result.
#
# Variable defaults come from dockerfile container
TVM_HOME=${TVM_HOME:="/src/tvm"}
ROCM_TVM_HOME=${ROCM_TVM_HOME:="/src/rocm-tvm"}

cd ${ROCM_TVM_HOME}/tutorials
python3 relay_quick_start.py
