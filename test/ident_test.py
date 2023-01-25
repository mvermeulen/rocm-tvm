# Basic example to run TVM on the identity ONNX graph

import onnx
import numpy as np
import tvm
import tvm.relay as relay

x = tvm.nd.array(np.random.uniform(size=(10)).astype('float32'))
print('inputs', x)

onnx_model = onnx.load('identity.onnx')

target = tvm.target.Target("rocm", host="llvm")
dev = tvm.rocm(0)

shape_dict = {'input': x.shape }
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=0):
    executor = relay.build_module.create_executor(
        "graph", mod, dev, target, params).evaluate()

tvm_output = executor(x)
print('outputs', tvm_output)
