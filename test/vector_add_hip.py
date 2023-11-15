# Basic example to run TVM on the identity ONNX graph

import onnx
import numpy as np
import tvm
import tvm.relay as relay

x = tvm.nd.array(np.random.uniform(size=(10)).astype('float32'))
y = tvm.nd.array(np.random.uniform(size=(10)).astype('float32'))
x2 = tvm.nd.array(np.random.uniform(size=(10)).astype('float16'))
y2 = tvm.nd.array(np.random.uniform(size=(10)).astype('float16'))
print('input1 ', x)
print('input2 ', y)

onnx_model = onnx.load('vector_add.onnx')

target = tvm.target.Target("hip")
dev = tvm.rocm(0)

shape_dict = {'input1': x.shape,
              'input2': y.shape }

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=0):
    executor = relay.build_module.create_executor(
        "graph", mod, dev, target, params).evaluate()

tvm_output = executor(x,y)
print('output', tvm_output)
print('\n')

onnx_model = onnx.load('vector_add_fp16.onnx')

shape_dict = {'input1': x2.shape,
              'input2': y2.shape }

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=0):
    executor = relay.build_module.create_executor(
        "graph", mod, dev, target, params).evaluate()

print('input1 ', x2)
print('input2 ', y2)    
tvm_output = executor(x2,y2)
print('output', tvm_output)
