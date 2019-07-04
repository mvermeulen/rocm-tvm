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
      
# Compile for CPU target
target='llvm'
madd_cpu =  tvm.build(s, [A,B,C], target, name='madd_cpu')

# Run the function in CPU context
ctx = tvm.cpu(0)
m = 512
n = 1024
a = tvm.nd.array(np.random.uniform(size=(m,n)).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=(m,n)).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(shape=(m,n),dtype=A.dtype), ctx)
madd_cpu(a,b,c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

print(madd_cpu.get_source())
print(madd_cpu.get_source('asm'))
