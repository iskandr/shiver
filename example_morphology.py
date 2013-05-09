import numpy as np 
import shiver 


_erode_compilation_cache = {} 
def erode(x, shape=(3,3)):
  
  x_elt_ctype = shiver.dtype_to_ctype_name(x.dtype) 
  m,n = shape 
  cache_key = (xt, m, n)
  if cache_key in _erode_compilation_cache:
    llvn_fn = _erode_compilation_cache[cache_key]
  else:
    src = """
      
  
  