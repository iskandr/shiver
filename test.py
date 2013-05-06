import numpy as np

import llvm 
import llvm.core 
import llvm.ee  
from llvm.ee import GenericValue 

import shiver
import testing_helpers 
import llvm_helpers
from llvm_helpers import  ty_int64,  ty_ptr_int64, empty_fn, ty_void, const 

global_module = llvm.core.Module.new("global_module")
ee = llvm.ee.ExecutionEngine.new(global_module)

def mk_identity_fn():
  fn = empty_fn(global_module, "ident", (ty_int64,), ty_int64  )
  bb = fn.append_basic_block("entry")
  builder = llvm.core.Builder.new(bb)
  builder.ret(fn.args[0])
  return fn 

ident = mk_identity_fn()

def test_should_fail():
  try:
    shiver.parfor(ident, niters=1)
    assert False, "Shouldn't have accepted function with non-void return"
  except:
    pass 

def test_add1_from_c():
  src = "int add1(int x) { return x + 1;}"
  add1 = llvm_helpers.from_c("add1", src);
  x = 1
  res = llvm_helpers.run(add1, x)
  y = res.as_int()
  assert (x+1) == y, "Expected %d but got %d" % (x+1,y)


"""
def mk_add1_to_elt():
  fn = empty_fn(global_module, "add1", 
                [("x", ty_ptr_int64), 
                 ("y", ty_ptr_int64),
                 ("i", ty_int64)
                ], 
                ty_void)
  bb = fn.append_basic_block("entry")
  builder = llvm.core.Builder.new(bb)
  x, y, i = fn.args 
  input_ptr = builder.gep(x, [i])
  input_elt = builder.load(input_ptr)
  output_elt = builder.add(input_elt,  const(1))
  output_ptr = builder.gep(y, [i])
  builder.store(output_elt, output_ptr)
  builder.ret_void()
  return fn 
"""
def mk_add1_to_elt():
  src = "void add1_to_elt(int* x, int* y, int i) { y[i] = x[i] + 1; }"
  return llvm_helpers.from_c("add1_to_elt", src, print_llvm = True)

add1_to_elt = mk_add1_to_elt()

def test_add1_arrays():   
  x = np.array([1,2,3,4])
  y = x.copy()
  shiver.parfor(add1_to_elt, len(x), fixed_args = (x, y), ee = ee)
  expected = x + 1
  assert all(y.shape == expected.shape)
  assert all(y == expected)

if __name__ == '__main__':
  testing_helpers.run_local_tests()