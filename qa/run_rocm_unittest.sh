#!/bin/bash
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
python3 test_codegen_rocm.py
python3 test_codegen_device.py
python3 test_codegen_bool.py
python3 test_lang_target.py
# expected output: "Testing using contexts: [cpu(0),rocm(0)]
python3 test_runtime_ndarray.py
# expected output: "Running on target: rocm", along with no errors
python3 test_lang_tensor_overload_op.py
