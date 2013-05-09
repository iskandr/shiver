 
from llvm.core import *  

from llvm_helpers import const_float, const_int, empty_fn, shared_module, optimize
from type_helpers import ty_pyobj, ty_ptr_pyobj, ty_void 

PyEval_InitThreads = \
  shared_module.add_function(Type.function(ty_void, []), "PyEval_InitThreads")
PyEval_SaveThread = \
  shared_module.add_function(Type.function(ty_ptr_pyobj, []), "PyEval_SaveThread")
PyEval_RestoreThread = \
  shared_module.add_function(Type.function(ty_void, [ty_ptr_pyobj]), "PyEval_RestoreThread")

PyThreadState_Swap = \
  shared_module.add_function(Type.function(ty_ptr_pyobj, [ty_ptr_pyobj]), "PyThreadState_Swap")

PyEval_ReleaseLock = \
  shared_module.add_function(Type.function(ty_void, []), "PyEval_ReleaseLock")
PyEval_AcquireLock = \
  shared_module.add_function(Type.function(ty_void, []), "PyEval_AcquireLock")


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

  
def mk_parfor_wrapper(fn, step_sizes):
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
  
  thread_state = entry_builder.call(PyThreadState_Swap, [null], 'thread_state')
  entry_builder.call(PyEval_ReleaseLock, [])
  
  exit_builder = loop_builder.create(entry_bb, entry_builder)
  
  exit_builder.call(PyEval_AcquireLock, [])
  exit_builder.call(PyThreadState_Swap, [thread_state])
  exit_builder.ret_void()
  
  # have to inline the call to the user function after the wrapper
  # is fully constructed or else LLVM crashes complaining 
  # about missing block terminators 
  loop_builder.inline_inner_call()
  return wrapper 

def parfor_wrapper(fn, step_sizes, _cache = {}):
  cache_key = id(fn), tuple(step_sizes) 
  if cache_key in _cache:
    return _cache[cache_key]
  else:
    wrapper = mk_parfor_wrapper(fn, step_sizes)
    optimize(wrapper)
    _cache[cache_key] = wrapper 
    print wrapper 
    return wrapper 
  
  