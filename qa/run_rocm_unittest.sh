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

cd ${TVM_HOME}/tests/python/unittest
# no output on success
python3 test_target_codegen_bool.py
python3 test_target_codegen_rocm.py
python3 test_target_codegen_device.py
python3 test_target_target.py
python3 test_te_tensor_overload.py
