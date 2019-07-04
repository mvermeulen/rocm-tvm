import tvm
import numpy as np

# Matrix add 
n = tvm.var('n')
m = tvm.var('m')
A = tvm.placeholder((m,n), name='A')
B = tvm.placeholder((m,n), name='B')
C = tvm.compute((m,n), lambda i,j: A[i,j] + B[i,j], name='C')

# Create default schedule
s = tvm.create_schedule(C.op)

# Example low level IR
print(tvm.lower(s, [A, B, C], simple_mode=True))

# Compile for GPU target
# GPU thread indices
block_x = tvm.thread_axis('blockIdx.x')
thread_x = tvm.thread_axis('threadIdx.x')

# split the workload
bx,tx = s[C].split(C.op.axis[0],factor=64)
print(tvm.lower(s,[A,B,C], simple_mode=True))
s[C].bind(bx,block_x)
s[C].bind(tx,thread_x)

target='rocm'
madd_gpu =  tvm.build(s, [A,B,C], target, target_host='llvm', name='madd_gpu')

# Run the function in GPU context
ctx = tvm.rocm(0)
m = 512
n = 1024
a = tvm.nd.array(np.random.uniform(size=(m,n)).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=(m,n)).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(shape=(m,n),dtype=A.dtype), ctx)
madd_gpu(a,b,c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

dev_module = madd_gpu.imported_modules[0]
print(dev_module.get_source('llvm'))
print(dev_module.get_source('asm'))

