import numpy as np

import llvm.core  

import shiver
import testing_helpers 
import llvm_helpers
from type_helpers import ty_int64
from llvm_helpers import empty_fn 


def mk_identity_fn():
  fn = empty_fn("ident", (ty_int64,), ty_int64 )
  bb = fn.append_basic_block("entry")
  builder = llvm.core.Builder.new(bb)
  builder.ret(fn.args[0])
  return fn 

ident = mk_identity_fn()

def test_return_type_causes_failure():
  try:
    shiver.parfor(ident, niters=1)
    assert False, "Shouldn't have accepted function with non-void return"
  except:
    pass 

def test_add1_from_c():
  add1 = llvm_helpers.from_c("add1", "int add1(int x) { return x + 1;}");
  x = 1
  res = shiver.run(add1, x)
  y = res.as_int()
  assert (x+1) == y, "Expected %d but got %d" % (x+1,y)



add1_src = "void add1_to_elt_int32(int* x, int* y, long i) { y[i] = x[i] + 1; }"
add1_to_elt_int32 = llvm_helpers.from_c("add1_to_elt_int32", add1_src)

def test_add1_arrays():
  n = 12    
  x = np.arange(n, dtype= np.int32)
  y = np.empty_like(x)
  shiver.parfor(add1_to_elt_int32, n, fixed_args = (x, y))
  expected = x + 1
  assert y.shape == expected.shape
  assert all(y == expected), "Expected %s but got %s" % (expected, y)

def test_input_type_error():
  n = 12 
  x = np.arange(n, dtype= np.int64)
  y = np.empty_like(x)
  # function expects int32* so call should fail 
  try: 
    shiver.parfor(add1_to_elt_int32, n, fixed_args = (x, y))
  except:
    return  
  assert False, "Shouldn't have succeeded due to int32*/int64* mismatch"

def test_timing():
  n = 10**8
  x = np.arange(n, dtype=float)
  y = np.empty_like(x)
  src = "void add1_to_elt_float64(double* x, double* y, long i) { y[i] = x[i] + 1.0; }"
  fn = llvm_helpers.from_c("add1_to_elt_float64", src)
  
  import time 
  start_t1 = time.time()
  shiver.parfor(fn, n, fixed_args = (x, y))
  stop_t1 = time.time()
  
  start_t2 = time.time()
  shiver.parfor(fn, n, fixed_args = (x, y))
  stop_t2 = time.time()
  
  start_t3 = time.time()
  np.add(x, 1.0, out=y)
  stop_t3 = time.time()
  print "First run time: %s" % (stop_t1 - start_t1)
  print "Second run time: %s" % (stop_t2 - start_t2) 
  print "Numpy time: %s" % (stop_t3 - start_t3)


if __name__ == '__main__':
  testing_helpers.run_local_tests()
