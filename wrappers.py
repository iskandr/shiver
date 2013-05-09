 
from llvm.core import *  

from llvm_helpers import const_float, const_int, empty_fn, shared_module, optimize
from llvm_helpers import return_type, input_types, input_names 
from type_helpers import ty_pyobj, ty_ptr_pyobj, ty_void , ty_int64
from type_helpers import is_llvm_float_type, is_llvm_int_type

def libfn(name, return_type, input_types):
  def get_ref(module = shared_module):
    t = Type.function(return_type, input_types)
    return module.get_or_insert_function(t, name)
  return get_ref 

PyEval_InitThreads = libfn("PyEval_InitThreads", ty_void, [])
PyEval_SaveThread = libfn("PyEval_SaveThread", ty_ptr_pyobj, [])
PyEval_RestoreThread = libfn("PyEval_RestoreThread", ty_void, [ty_ptr_pyobj])
PyThreadState_Swap = libfn("PyThreadState_Swap", ty_ptr_pyobj, [ty_ptr_pyobj])
PyEval_ReleaseLock = libfn ("PyEval_ReleaseLock", ty_void, [])
PyEval_AcquireLock = libfn ("PyEval_AcquireLock", ty_void, [])


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
    self.inner_call = None 
    
  _loop_var_names = ['i', 'j', 'k', 'l']
  
  def new_block(self, name):
    bb = self.wrapper.append_basic_block(name)
    builder = Builder.new(bb)
    return bb, builder
  
  def get_var_name(self, pos):
    maxpos = len(self._loop_var_names)
    return self._loop_var_names[pos % maxpos] * ((pos/maxpos)+1)
  
  def inline_inner_call(self):
    """
    After the wrapper function is fully constructed, 
    call this method to inline the user function into 
    the innermost loop nest
    """
    llvm.core.inline_function(self.inner_call)
    
  def create(self, bb, builder, loop_idx_values = []):
    n = len(loop_idx_values)
    if n == self.n_loops:
      # save the LLVM call node so we can inline it later 
      self.inner_call = builder.call(self.original_fn, self.closure_values + loop_idx_values)
     
      
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
      body_builder = self.create(body_bb, body_builder, loop_idx_values + [idx_value])
      step = const_int(self.step_constants[n], idx_value.type)
      next_idx_value = body_builder.add(idx_value, step)
      body_builder.store(next_idx_value, var)
      body_builder.branch(test_bb)
      return after_builder 

  
def mk_parfor_wrapper_no_return(fn, step_sizes):
  n_indices = len(step_sizes)

  # need to double the integer index inputs to allow for a stop argument 
  # of each   the number of index inputs to
  old_input_types = [arg.type for arg in fn.args]
  new_input_types = [t for t in old_input_types]
  for i in xrange(n_indices):
    old_index_type = old_input_types[-n_indices + i]
    assert old_index_type.kind == TYPE_INTEGER, \
        "Last %d input(s) to %s must be integers" % (n_indices, fn.name)
    new_input_types.append(old_index_type)
  wrapper = empty_fn(fn.name + "_wrapper",  new_input_types, module = fn.module)
   
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
  
  
  entry_bb, entry_builder = loop_builder.new_block("entry")
  
  
  null = llvm.core.Constant.null(ty_ptr_pyobj)
  
  thread_state = entry_builder.call(PyThreadState_Swap(fn.module), [null], 'thread_state')
  entry_builder.call(PyEval_ReleaseLock(fn.module), [])
  
  exit_builder = loop_builder.create(entry_bb, entry_builder)
  
  exit_builder.call(PyEval_AcquireLock(fn.module), [])
  exit_builder.call(PyThreadState_Swap(fn.module), [thread_state])
  exit_builder.ret_void()
  
  # have to inline the call to the user function after the wrapper
  # is fully constructed or else LLVM crashes complaining 
  # about missing block terminators 
  loop_builder.inline_inner_call()
  return wrapper 

def mk_parfor_wrapper_collect_returned_values(fn, step_sizes, dim_sizes, result_t):
  
  name = "write_output_%s" % fn.name 
  for step_size in step_sizes:
    name = name + "_%d" % step_size
  write_output_fn = empty_fn(name, 
                     [Type.pointer(result_t)] + input_types(fn), 
                     output_type=ty_void, 
                     module = fn.module)
   
  original_args = write_output_fn.args[1:]
  bb = write_output_fn.append_basic_block("entry")
  builder = Builder.new(bb)
  value = builder.call(fn, original_args, "elt_result")
  
  n_indices = len(step_sizes)
  idx_args = write_output_fn.args[-n_indices:]
  strides = [1]
  for dim_size in reversed(dim_sizes[1:]):
    strides.append(strides[-1] * dim_size)
  strides.reverse()
  output_idx = const_int(0, ty_int64)
  for (idx, stride) in zip(idx_args, strides):
    dim_offset = builder.mul(idx, const_int(stride, idx.type))
    output_idx = builder.add(output_idx, dim_offset)
  output_array = write_output_fn.args[0]
  output_ptr = builder.gep(output_array, [output_idx])
  builder.store(value, output_ptr)
  builder.ret_void()
  llvm.core.inline_function(value)
  return mk_parfor_wrapper_no_return(write_output_fn, step_sizes)
   

def parfor_wrapper(fn, step_sizes, dim_sizes = None, _cache = {}):
  cache_key = id(fn), tuple(step_sizes) 
  if cache_key in _cache:
    return _cache[cache_key]
  else:
    result_t = fn.type.pointee.return_type 
    if result_t == ty_void:
      wrapper = mk_parfor_wrapper_no_return(fn, step_sizes)
    else:
      assert is_llvm_float_type(result_t) or is_llvm_int_type(result_t), \
         "Function can't return type %s" % result_t
      assert dim_sizes is not None 
      wrapper = mk_parfor_wrapper_collect_returned_values(fn, step_sizes, dim_sizes, result_t)
    optimize(wrapper)
    _cache[cache_key] = wrapper 

    return wrapper 
  
  