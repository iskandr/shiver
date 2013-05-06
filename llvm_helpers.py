import numpy as np


from llvm import * 
from llvm.core import * 

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
      builder.call(self.original_fn, self.closure_values + loop_vars)
      return builder
    else:
      var = builder.alloca(ty_int64, self.get_var_name(n))
     
      builder.store(self.start_values[n],var)
      test_bb, test_builder =  self.new_block("test%d" % (n+1))
      builder.branch(test_bb)
      idx_value = test_bb.load(var)                         
      cond = builder.icmp(ICMP_ULT, idx_value,self.stop_values[n], "stop_cond%d" % (n+1))
      body_bb, body_builder = self.new_block("body%d" % (n+1))
      after_bb, after_builder = self.new_block("after_loop%d" % (n+1))
      test_builder.cbranch(cond, body_bb, after_bb)
      body_builder = self.create(body_bb, body_builder, loop_vars + [var])
      body_builder.branch(test_builder)
      return after_builder 
    
def empty_fn(module, name, input_types, output_type = ty_void):
  ty_func = Type.function(output_type, input_types)
  return module.add_function(ty_func, name)  

def mk_wrapper(fn, step_sizes):
  n_indices = len(step_sizes)

  # need to double the integer index inputs to allow for a stop argument 
  # of each   the number of index inputs to
  old_input_types = [arg.type for arg in fn.args]
  extra_input_types = [ty_int64 for _ in xrange(n_indices)]
  new_input_types = old_input_types + extra_input_types
  wrapper = empty_fn(fn.module, fn.name + "_wrapper",  new_input_types)
  
  for arg_idx in xrange(len(new_input_types)):
    if arg_idx < len(old_input_types):
      wrapper.args[arg_idx].name = fn.args[arg_idx].name 
    else:
      wrapper.args[arg_idx].name = fn.args[arg_idx - n_indices].name + "_stop"
  index_vars = wrapper.args[-2*n_indices:]
  closure_vars = wrapper.args[:-2*n_indices]
  start_vars = index_vars[:n_indices]
  stop_vars = index_vars[n_indices:]
  
  assert len(start_vars) == len(stop_vars)
  loop_builder = LoopBuilder(wrapper, 
                             fn, 
                             closure_vars, 
                             start_vars, 
                             stop_vars, 
                             step_sizes)
  exit_builder = loop_builder.create()
  exit_builder.ret()
  return wrapper 