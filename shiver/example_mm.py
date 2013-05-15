"""
Compile a different matrix-multiply kernel for every distinct input 
shape and run these kernels in parallel with shiver.
"""

import numpy as np 
import shiver 

_compilation_cache = {}
def mm(x, y, output_elt_type = None):
  m,d = x.shape 
  n = y.shape[1]
  assert d == y.shape[0]
  
  # use NumPy's type logic to figure out the 
  # sanest type to contain the products of elements
  # from x and y 
  if output_elt_type is None: 
    output_elt_type = np.promote_types(x.dtype, y.dtype)
    
  xt = shiver.dtype_to_ctype_name(x.dtype)
  yt = shiver.dtype_to_ctype_name(y.dtype)
  zt = shiver.dtype_to_ctype_name(output_elt_type)  
  
  cache_key = (m,n,d,xt,yt,zt)

  if cache_key in _compilation_cache:
    llvm_fn = _compilation_cache[cache_key]
  else:
    name = "mm_%(m)d_%(n)d_%(d)d_%(xt)s_%(yt)s_%(zt)s" % locals()
    args = "%(xt)s* x, %(yt)s* y, long i, long j" % locals()
    src = """
      %(zt)s %(name)s (%(args)s) {     
        long m = %(m)d; 
        long n = %(n)d; 
        long d = %(d)d; 
        %(zt)s total = 0;
        
        for (long k = 0; k < d; ++k) { 
          // assume row-major indexing on 
          // both input arrays, which should 
          // be ensured by the calling code 
          %(xt)s xelt = x[i*d + k];
          %(yt)s yelt = y[j*d + k];
          total += xelt * yelt;   
        }
        return total;
      }
    """ % locals()

    llvm_fn = shiver.from_c(src)
    _compilation_cache[cache_key] = llvm_fn
   
  if not x.flags.c_contiguous: x = x.copy()
  y = y.T 
  if not y.flags.c_contiguous: y = y.copy()
  return shiver.parfor(llvm_fn, niters=(m,n), fixed_args=[x,y])
  

import time 
if __name__ == '__main__':
  m, n, d = 1000,1000,1000
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
  
  
    