import numpy as np

import llvm 
import llvm.core 
import llvm.ee as ee 

import shiver
import llvm_helpers
from llvm_helpers import  ty_int64,  ty_ptr_int64, empty_fn, ty_void, const 

global_module = llvm.core.Module.new("global_module")

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
  
def mk_add1():
  fn = empty_fn(global_module, "add1", [ty_ptr_int64, ty_ptr_int64, ty_int64], ty_void)
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
  
add1 = mk_add1()

def test_add1():
  x = np.array([1,2,3,4])
  y = x.copy()
  x_ptr = ee.GenericValue.pointer(x.ctypes.data)
  y_ptr = ee.GenericValue.pointer(y.ctypes.data)
  shiver.parfor(add1, len(x), fixed_args= (x_ptr, y_ptr))
  expected = x + 1
  assert all(y.shape == expected.shape)
  assert all(y == expected)