import numpy as np
a = np.uint8(-5.0)
b = np.float32(a)
c = np.float16(b)
d = np.int8(c)
print(a.dtype, a)
print(b.dtype, b)
print(c.dtype, c)
print(d.dtype, d)