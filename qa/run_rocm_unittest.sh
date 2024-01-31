#!/bin/bash
#set -x
#
# RUN TVM ROCm code generation unit test
#
# Expected output: see below
#
# Variable defaults come from dockerfile container
TVM_HOME=${TVM_HOME:="/src/tvm"}
ROCM_TVM_HOME=${ROCM_TVM_HOME:="/src/rocm-tvm"}

cd ${TVM_HOME}/tests/python/

find . -type f -exec grep -li rocm {} \; | while read test
do
    echo --- $test ---
    python3 $test
done
