"""
Compile a different matrix-multiply kernel for every distinct input 
shape and run these kernels in parallel with shiver.
"""

import numpy as np 
import shiver 

_compilation_cache = {}
def mm(x, y, output_elt_type = None):
  
  xt = x.dtype 
  yt = y.dtype
  if output_elt_type is None:
    # use NumPy's type logic to figure out the 
    # sanest type to contain the products of elements
    # from x and y 
    zt = np.promote_types(xt, yt)
  else:
    zt = output_elt_type
  m,d = x.shape 
  n = y.shape[1]
  assert d == y.shape[0]
  
 
  
  cache_key = (m,n,d,xt,yt,output_elt_type)
  
  if cache_key in _compilation_cache:
    llvm_fn = _compilation_cache[cache_key]
  else:
    x_ctype = shiver.dtype_to_ctype_name(xt)
    y_ctype = shiver.dtype_to_ctype_name(yt)
    z_ctype = shiver.dtype_to_ctype_name(zt)
    name = "mm_%(m)d_%(n)d_%(d)d_%(xt)s_%(yt)s_%(zt)s" % locals()
    args = "%(x_ctype)s* x, %(y_ctype)s* y, %(z_ctype)s* z, long i, long j" % locals()
    
    body = """
       int m = %(m)d; 
       int n = %(n)d; 
       int d = %(d)d; 
       %(z_ctype)s total = 0;  
       for (int k = 0; k < d; ++k) { 
         // assume row-major indexing on 
         // both input arrays, which should 
         // be ensured by the calling code 
         %(x_ctype)s xelt = x[i*d + k];
         %(y_ctype)s yelt = y[j*d + k];
         total += xelt * yelt;   
       }
       z[i*n + j] = total;
    """ % locals()
    src = "void %(name)s (%(args)s) { %(body)s }" % locals() 
    print src 
    llvm_fn = shiver.from_c(name, src)
    _compilation_cache[cache_key] = llvm_fn
   
  if not x.flags.c_contiguous: x = x.copy()
  y = y.T 
  if not y.flags.c_contiguous: y = y.copy()
  z = np.empty((m,n), dtype = zt)
  shiver.parfor(llvm_fn, niters=(m,n), fixed_args=[x,y,z])
  return z

import time 
if __name__ == '__main__':
  m, n, d = 2000,1000,200
  x = np.random.randn(m,d)
  y = np.random.randn(d,n)
  
  comp_time_start = time.time()
  z = mm(x, y)
  comp_time_end = time.time()
  
  print "Shiver (w/ compilation time): %0.3f" % (comp_time_end - comp_time_start)
  
  # sanity check the output to make sure we're actually doing a matrix mult 
  np_time_start = time.time()
  expected = np.dot(x,y)
  np_time_end = time.time()
  print "NumPy time: %0.3f" % (np_time_end - np_time_start)
  
  nocomp_time_start = time.time()
  z = mm(x, y)
  nocomp_time_end = time.time()
  print "Shiver (w/out compilation) time: %0.3f" % (nocomp_time_end - nocomp_time_start)
  
  
  
  
  assert np.all( (z - expected) ** 2 < 0.00001)
  
  
    