import tvm
from tvm import te
import tvm.testing
import numpy as np

# Matrix add 
n = te.var('n')
A = te.placeholder((n), name='A')
B = te.placeholder((n), name='B')
C = te.compute((n), lambda i: A[i] + B[i], name='C')

# Create default schedule
s = te.create_schedule(C.op)

# Example low level IR
print(tvm.lower(s, [A, B, C], simple_mode=True))

# Compile for GPU target
# GPU thread indices
block_x = te.thread_axis('blockIdx.x')
thread_x = te.thread_axis('threadIdx.x')

# split the workload
bx,tx = s[C].split(C.op.axis[0],factor=64)
print(tvm.lower(s,[A,B,C], simple_mode=True))
s[C].bind(bx,block_x)
s[C].bind(tx,thread_x)

target='rocm'
vadd_gpu =  tvm.build(s, [A,B,C], target, target_host='llvm', name='vadd_gpu')

# Run the function in GPU context
ctx = tvm.rocm(0)
n = 20
a = tvm.nd.array(np.random.uniform(size=(n)).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=(n)).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(shape=(n),dtype=A.dtype), ctx)
vadd_gpu(a,b,c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

dev_module = vadd_gpu.imported_modules[0]
print(dev_module.get_source('llvm'))
print(dev_module.get_source('asm'))

