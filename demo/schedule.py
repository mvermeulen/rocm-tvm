import tvm
import numpy as np

# Matrix addition
n = tvm.var('n')
m = tvm.var('m')

A = tvm.placeholder((m,n), name='A')
B = tvm.placeholder((m,n), name='B')
C = tvm.compute((m,n), lambda i,j: A[i,j] + B[i,j], name='C')

s = tvm.create_schedule([C.op])
print(tvm.lower(s, [A,B,C], simple_mode=True))

# Split a specified axis
C = tvm.compute((m,n), lambda i,j: A[i,j] + B[i,j], name='C')
s = tvm.create_schedule([C.op])

x0, x1 = s[C].split(C.op.axis[0],factor=64)
print(tvm.lower(s, [A,B,C], simple_mode=True))

# Tile over two axis
C = tvm.compute((m,n), lambda i,j: A[i,j] + B[i,j], name='C')
s = tvm.create_schedule([C.op])
x0, y0, x1, y1 = s[C].tile(C.op.axis[0],C.op.axis[1], x_factor=10, y_factor=5)
print(tvm.lower(s, [A,B,C], simple_mode=True))

# Fuse over two consecutive axis
C = tvm.compute((m,n), lambda i,j: A[i,j] + B[i,j], name='C')
s = tvm.create_schedule([C.op])
x0, y0, x1, y1 = s[C].tile(C.op.axis[0],C.op.axis[1], x_factor=10, y_factor=5)
fused = s[C].fuse(x1,y1)
print(tvm.lower(s, [A,B,C], simple_mode=True))

# Reorder axis
C = tvm.compute((m,n), lambda i,j: A[i,j] + B[i,j], name='C')
s = tvm.create_schedule([C.op])
x0, y0, x1, y1 = s[C].tile(C.op.axis[0],C.op.axis[1], x_factor=10, y_factor=5)
s[C].reorder(x1,y0,x0,y1)
print(tvm.lower(s, [A,B,C], simple_mode=True))
