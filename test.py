import numpy as np

import llvm 
import llvm.core 
import llvm.ee as ee 

import shiver
import llvm_helpers
from llvm_helpers import  ty_int64, empty_fn 

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