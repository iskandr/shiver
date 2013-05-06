import numpy as np

import llvm 
import llvm.core 
import llvm.ee  
from llvm.ee import GenericValue 

import shiver
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
  x_gv = GenericValue.int(ty_int64, x) 
  res = ee.run_function(add1, [x_gv])
  y = res.as_int()
  assert (x+1) == y, "Expected %d but got %d" % (x+1,y)
    
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
  
add1_to_elt = mk_add1_to_elt()

"""
def test_add1_arrays():
  x = np.array([1,2,3,4])
  y = x.copy()
  x_ptr = GenericValue.pointer(x.ctypes.data)
  y_ptr = GenericValue.pointer(y.ctypes.data)
  shiver.parfor(add1_to_elt, len(x), fixed_args= (x_ptr, y_ptr))
  expected = x + 1
  assert all(y.shape == expected.shape)
  assert all(y == expected)
"""