import numpy as np 
import shiver 


def memoize(fn):
  cache = {} 
  def wrapped_fn(x, window_shape = (3,3)):
    key = x.dtype, window_shape
    if key in cache:
      return cache[key]
    src = fn(x, window_shape)
    cache[key] = src
    return src 
  return wrapped_fn 
    
@memoize
def build_erode(x, window_shape):
  xtype = shiver.dtype_to_ctype_name(x.dtype)
  m,n = window_shape
  return """ 
    %(xtype)s erode_%(xtype)s_%(m)d_%(n)d (%(xtype)s* x, long nrows, long ncols, long i, long j) { 
      
    }
  """ % locals()
    
      
      
  
  