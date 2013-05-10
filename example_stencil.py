import numpy as np 
import shiver 



def build_stencil_src(prefix = "erode", 
                        elt_type = 'double', 
                        window_shape = (3,3),
                        initial_value = 'x_ij', 
                        reduce_expr = 'result < x_ij ? result : x_ij',
                        final_expr = 'result'):
  m,n = window_shape
  hm = m/2 
  hn = n/2   
  fn_name = "%(prefix)s_%(elt_type)s_%(m)d_%(n)d" % locals()
  args = "%(elt_type)s* x, long nrows, long ncols, long i, long j" % locals()
 
  return """
    %(elt_type)s  %(fn_name)s(%(args)s) {
      long start_i = i -  %(hm)d;
      long start_j = j - %(hn)d;
      long stop_i = i + %(hm)d + 1;
      long stop_j = j + %(hn)d + 1;
      int border = 0; 
      %(elt_type)s x_ij = x[i*ncols+j];
      %(elt_type)s result = %(initial_value)s;
      if (start_i < 0) { start_i = 0; border = 1; } 
      if (start_j < 0) { start_j = 0; border = 1;  }
      if (stop_i > nrows) { stop_i = nrows; border = 1;  }
      if (stop_j > ncols) { stop_j = ncols; border = 1; }
      for (long i = start_i; i < stop_i; ++i) { 
        for (long j = start_j; j < stop_j; ++j) {
          x_ij = x[i*ncols+j];
          result = %(reduce_expr)s; 
        }
      }
      return %(final_expr)s;
    }  
    """ % locals()
  
def build_stencil(prefix = "erode", 
                    elt_type = 'double', 
                    window_shape = (3,3),
                    initial_value = 'upper_left', 
                    reduce_expr = 'result < x_ij ? result : x_ij', 
                    final_expr = 'result',
                    _cache = {}):
  key = (prefix, elt_type, window_shape, initial_value, reduce_expr, final_expr)
  if key in _cache:
    return _cache[key]
  else:
    src = build_stencil_src(prefix, elt_type, window_shape, initial_value, reduce_expr)
    llvm_fn = shiver.from_c(src)
    _cache[key] = llvm_fn 
    return llvm_fn 
   

def erode(x, window_shape = (3,3)):
  xt = shiver.dtype_to_ctype_name(x.dtype)
  llvm_fn = build_stencil(prefix = "erode", 
                          elt_type = xt, 
                          window_shape = window_shape, 
                          initial_value="x_ij", 
                          reduce_expr="result < x_ij ? result : x_ij")
  return shiver.parfor(llvm_fn, x.shape, fixed_args=[x, x.shape[0], x.shape[1]])

def dilate(x, window_shape = (3,3)):
  xt = shiver.dtype_to_ctype_name(x.dtype)
  llvm_fn = build_stencil(prefix = "dilate", 
                          elt_type = xt, 
                          window_shape = window_shape, 
                          initial_value="x_ij", 
                          reduce_expr="result > x_ij ? result : x_ij")
  return shiver.parfor(llvm_fn, x.shape, fixed_args=[x, x.shape[0], x.shape[1]])

def avg(x, window_shape = (3,3)):
  xt = shiver.dtype_to_ctype_name(x.dtype)
  llvm_fn = build_stencil(prefix = "avg", 
                          elt_type = xt, 
                          window_shape = window_shape, 
                          initial_value="0", 
                          reduce_expr="result + x_ij", 
                          final_expr = "result / (%d*%d)" % window_shape)
  return shiver.parfor(llvm_fn, x.shape, fixed_args=[x, x.shape[0], x.shape[1]])

import pylab 
from PIL import Image 
import scipy.ndimage  

if __name__ == '__main__':
  dali_jpeg = Image.open('dali.jpg')
  rgb = np.array(dali_jpeg)
  g = rgb[:,:,1] / 256.0
  import time 
  t1 = time.time()
  ge = avg(g)
  t2 = time.time()
  ge = erode(dilate(avg(g), (1,3)), (20,1))
  t3 = time.time()
  ge2 = scipy.ndimage.grey_erosion(g, (3,3))
  t4 = time.time()
  
  print "With compilation: %0.3f" % (t2 - t1)
  print "W/out compilation: %0.3f" % (t3 - t2)
  print "SciPY time: %0.3f" % (t4 - t3)
  #assert (ge == ge2).all() 
  pylab.imshow(np.dstack([ge,ge,ge]))
  pylab.figure()
  pylab.imshow(np.dstack([ge2,ge2,ge2]))
  pylab.show() 
  