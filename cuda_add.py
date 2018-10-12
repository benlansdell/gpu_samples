#!python

#Based on https://wiki.tiker.net/PyCuda/Examples
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

#Size of array
N = 222341

block_size = 256

x = np.ones(N)
y = 2*np.ones(N)

func_mod = SourceModule("""
extern "C" {
__global__ void func(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
}
""", no_extern_c=1)

func = func_mod.get_function('func')
x = np.asarray(x, np.float32)
x_gpu = gpuarray.to_gpu(x)
y = np.asarray(y, np.float32)
y_gpu = gpuarray.to_gpu(y)

# a function to the GPU
func(np.uint32(N), x_gpu.gpudata, y_gpu.gpudata, block=(block_size, 1, 1), grid=(1,1,1))
cuda.Context.synchronize()

result_cpu = x + y
result_gpu = y_gpu.get()

print 'x+y:       ', y[N-1]+x[N-1]
#Retrieve result from GPU. returns a numpy array
print 'func(x,y): ', y_gpu.get()[N-1]
x_colors=x_gpu.get()

print 'max error', np.max(result_cpu - result_gpu)