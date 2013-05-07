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

to_lltype_mappings = {
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

to_numpy_type_mappings = {
  str(ty_int8) : np.int8,              
  str(ty_int16) : np.int16, 
  str(ty_int32) : np.int32,  
  str(ty_int64) : np.int64,  
  str(ty_float32) : np.float32,  
  str(ty_float64) : np.float64,
}

to_dtype_mappings = {}
for (k,v) in to_numpy_type_mappings.iteritems():
  to_dtype_mappings[k] = np.dtype(v)

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
  assert t in to_lltype_mappings, "Unknown type %s" % (t,)
  return to_lltype_mappings[t]


def return_type(fn):
  return fn.type.pointee.return_type 


def const_int(x, t = ty_int64):
  return Constant.int(t, x)

def const_float(x, t = ty_float64):
  return Constant.real(t, x)

def const(x):
  if isinstance(x, int):
    return const_int(x)
  else:
    assert isinstance(x, float)
    return const_float(x)

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
  
  def _create(self, bb, builder, loop_idx_values = []):
    n = len(loop_idx_values)
    if n == self.n_loops:
      builder.call(self.original_fn, self.closure_values + loop_idx_values)
      return builder
    else:
      start = self.start_values[n]
      stop = self.stop_values[n]
      var = builder.alloca(start.type, self.get_var_name(n))
  
      builder.store(start, var)
      test_bb, test_builder =  self.new_block("test%d" % (n+1))
      builder.branch(test_bb)
      idx_value = test_builder.load(var)                         
      cond = test_builder.icmp(ICMP_ULT, idx_value, stop, "stop_cond%d" % (n+1))
      body_bb, body_builder = self.new_block("body%d" % (n+1))
      after_bb, after_builder = self.new_block("after_loop%d" % (n+1))
      test_builder.cbranch(cond, body_bb, after_bb)
      body_builder = self._create(body_bb, body_builder, loop_idx_values + [idx_value])
      step = const_int(self.step_constants[n], idx_value.type)
      next_idx_value = body_builder.add(idx_value, step)
      body_builder.store(next_idx_value, var)
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

def is_llvm_float_type(t):
  return t.kind in (llvm.core.TYPE_FLOAT, llvm.core.TYPE_DOUBLE)

def is_llvm_int_type(t):
  return t.kind == llvm.core.TYPE_INTEGER

def is_llvm_ptr_type(t):
  return t.kind == llvm.core.TYPE_POINTER

def from_python(x, llvm_type = None):
  if isinstance(x, GenericValue):
    return x
  elif isinstance(x, (int,long)):
    llvm_type =  ty_int64 if llvm_type is None else llvm_type
    assert is_llvm_int_type(llvm_type), \
      "Expected LLVM integer type, not %s" % (llvm_type,) 
    return GenericValue.int(llvm_type, x)
  elif isinstance(x, float):
    llvm_type = ty_float64 if llvm_type is None else llvm_type
    assert is_llvm_float_type(llvm_type), \
        "Expected LLVM float type, not %s" % (llvm_type,)
    return GenericValue.real(llvm_type, x)  
  elif isinstance(x, bool):
    llvm_type = ty_int8 if llvm_type is None else llvm_type
    assert is_llvm_int_type(llvm_type), \
      "Expected LLVM integer type, not %s" % (llvm_type,)
    return GenericValue.int(llvm_type, x)
  else:
    
    assert isinstance(x, np.ndarray)
    assert llvm_type is not None 
    assert is_llvm_ptr_type(llvm_type), \
      "Native argument receiving numpy array must be a pointer, not %s" % (llvm_type,)
    elt_type = llvm_type.pointee
    assert is_llvm_float_type(elt_type) or is_llvm_int_type(elt_type)
    elt_type_str = str(elt_type) 
    assert elt_type_str in to_dtype_mappings, \
      "Don't know how to convert LLVM type %s to dtype" % (elt_type_str,)
    dtype = to_dtype_mappings[elt_type_str]
    assert dtype == x.dtype, \
        "Can't pass array with %s* data pointer to function that expects %s*" % (x.dtype, dtype)
    return GenericValue.pointer(x.ctypes.data)
  

def run(llvm_fn, *input_values, **kwds):
  """
  Given a compiled LLVM function and Python input values, 
  convert the input to LLVM generic values and run the 
  function 
  """
  ee = kwds.get('ee')
  llvm_inputs = [from_python(v, arg.type) 
                 for (v,arg) in 
                 zip(input_values, llvm_fn.args)]
  if ee is None:
    ee = ExecutionEngine.new(llvm_fn.module)
  return ee.run_function(llvm_fn, llvm_inputs)