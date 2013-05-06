import numpy as np


from llvm import * 
from llvm.core import *
from llvm.ee import GenericValue, ExecutionEngine 

ty_void = Type.void()
ty_int8 = Type.int(8)
ty_int16 = Type.int(16) 
ty_int32 = Type.int(32) 
ty_int64 = Type.int(64) 

ty_float32 = Type.float()
ty_float64 = Type.double()

ty_ptr_int8 = Type.pointer(ty_int8)
ty_ptr_int16 = Type.pointer(ty_int16)
ty_ptr_int32 = Type.pointer(ty_int32)
ty_ptr_int64 = Type.pointer(ty_int64)

ty_ptr_float32 = Type.pointer(ty_float32)
ty_ptr_float64 = Type.pointer(ty_float64)

mappings = {
  np.int8 : ty_int8,             
  np.int16 : ty_int16, 
  np.int32 : ty_int32, 
  np.int64 : ty_int64, 
  np.float32 : ty_float32, 
  np.float64 : ty_float64,
  int : ty_int64, 
  float : ty_float64, 
  bool : ty_int8, 
}

def ptr(t):
  return Type.pointer(t)

def to_lltype(t):
  """
  Convert python types to LLVM types.
  Examples: 
    to_lltype(int) == Type.int(64)
    to_lltype([float]) == Type.pointer(Type.double())
  """
  if isinstance(t, np.dtype):
    return to_lltype(t.type)
  elif isinstance(t, (list,tuple)):
    assert len(t) == 1
    elt_t = to_lltype(t[0])
    return ptr(elt_t) 
  assert t in mappings, "Unknown type %s" % (t,)
  return mappings[t]


def return_type(fn):
  return fn.type.pointee.return_type 


def int_const(x, t = ty_int64):
  return Constant.int(t, x)

def float_const(x, t = ty_float64):
  return Constant.real(t, x)

def const(x):
  if isinstance(x, int):
    return int_const(x)
  else:
    assert isinstance(x, float)
    return float_const(x)

class LoopBuilder(object):
  def __init__(self, wrapper, original_fn, closure_values, 
               start_values, stop_values, step_constants):
    self.wrapper = wrapper 
    self.original_fn = original_fn 
    self.closure_values = closure_values 
    self.start_values = start_values 
    self.stop_values = stop_values 
    self.step_constants = step_constants 
    self.n_loops = len(start_values)
    assert len(stop_values) == self.n_loops 
    assert len(step_constants) == self.n_loops 
    n_args = len(original_fn.args)
    assert len(closure_values) + self.n_loops == n_args
  
  _loop_var_names = ['i', 'j', 'k', 'l']
  
  def new_block(self, name):
    bb = self.wrapper.append_basic_block(name)
    builder = Builder.new(bb)
    return bb, builder
  
  def get_var_name(self, pos):
    maxpos = len(self._loop_var_names)
    return self._loop_var_names[pos % maxpos] * ((pos/maxpos)+1)
  
  def create(self):
    bb, builder = self.new_block("entry")
    return self._create(bb,builder)
  
  def _create(self, bb, builder, loop_vars = []):
    n = len(loop_vars)
    if n == self.n_loops:
      loop_idx_values = [builder.load(var) for var in loop_vars]
      builder.call(self.original_fn, self.closure_values + loop_idx_values)
      return builder
    else:
      var = builder.alloca(ty_int64, self.get_var_name(n))
     
      builder.store(self.start_values[n],var)
      test_bb, test_builder =  self.new_block("test%d" % (n+1))
      builder.branch(test_bb)
      idx_value = test_builder.load(var)                         
      cond = builder.icmp(ICMP_ULT, idx_value,self.stop_values[n], "stop_cond%d" % (n+1))
      body_bb, body_builder = self.new_block("body%d" % (n+1))
      after_bb, after_builder = self.new_block("after_loop%d" % (n+1))
      test_builder.cbranch(cond, body_bb, after_bb)
      body_builder = self._create(body_bb, body_builder, loop_vars + [var])
      body_builder.branch(test_bb)
      return after_builder 

  

def empty_fn(module, name, input_types, output_type = ty_void):
  names = []
  types = []
  for (i, item) in enumerate(input_types):
    if isinstance(item, (list, tuple)):
      names.append(item[0])
      types.append(item[1])
    else:
      names.append("arg%d" % (i+1))
      types.append(item)
  
  ty_func = Type.function(output_type, types)
  fn = module.add_function(ty_func, name)
  for (i,arg) in enumerate(fn.args):
    arg.name = names[i]
  return fn  

import subprocess 
import tempfile
def from_c(fn_name, src, compiler = 'clang', print_llvm = False):
  src_filename = tempfile.mktemp(prefix = fn_name + "_src_", suffix = '.c')
  print src_filename
  f = open(src_filename, 'w')
  f.write(src + '\n')
  f.close()
  if print_llvm:
    assembly_filename = tempfile.mktemp(prefix = fn_name + "_llcode_", suffix = '.s')
    subprocess.check_call([compiler,  '-c', '-emit-llvm', '-S',  src_filename, '-o', assembly_filename, ])
    llvm_source = open(assembly_filename).read()
    print llvm_source
    module = Module.from_assembly(llvm_source)
    
  else: 
    bitcode_filename = tempfile.mktemp(prefix = fn_name + "_bitcode_", suffix = '.o')
    subprocess.check_call([compiler,  '-c', '-emit-llvm',  src_filename, '-o', bitcode_filename, ])
    module = Module.from_bitcode(open(bitcode_filename))
  return module.get_function_named(fn_name)

def from_python(x):
  if isinstance(x, (int,long)):
    return GenericValue.int(ty_int64, x)
  elif isinstance(x, float):
    pass 
  elif isinstance(x, bool):
    return GenericValue.int(ty_int8, x)
  else:
    assert isinstance(x, np.ndarray), \
      "Don't know how to convert Python value of type %s" % (type(x),)
    return GenericValue.pointer(x.ctypes.data)
  

def run(llvm_fn, *input_args, **kwds):
  """
  Given a compiled LLVM function and Python input values, 
  convert the input to LLVM generic values and run the 
  function 
  """
  ee = kwds.get('ee')
  llvm_inputs = [from_python(x) for x in input_args]
  if ee is None:
    ee = ExecutionEngine.new(llvm_fn.module)
  return ee.run_function(llvm_fn, llvm_inputs)