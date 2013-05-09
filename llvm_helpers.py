

import subprocess 
import tempfile

from llvm import * 
from llvm.core import *
from llvm.ee import GenericValue, ExecutionEngine 
from type_helpers import ty_int64, ty_float64, ty_void, lltype_to_ctype, ty_ptr_pyobj,\
  ty_pyobj


shared_module = llvm.core.Module.new("shiver_global_module")


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


  
shared_exec_engine = llvm.ee.ExecutionEngine.new(shared_module)

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
  
  
  def create(self, bb, builder, loop_idx_values = []):
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
      body_builder = self.create(body_bb, body_builder, loop_idx_values + [idx_value])
      step = const_int(self.step_constants[n], idx_value.type)
      next_idx_value = body_builder.add(idx_value, step)
      body_builder.store(next_idx_value, var)
      body_builder.branch(test_bb)
      return after_builder 

  

def empty_fn( name, input_types, output_type = ty_void, module = shared_module):
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


def module_from_c(src, name = 'fn', compiler = 'clang', print_llvm = False):
  src_filename = tempfile.mktemp(prefix = name + "_src_", suffix = '.c')

  f = open(src_filename, 'w')
  f.write(src + '\n')
  f.close()
  if print_llvm:
    assembly_filename = tempfile.mktemp(prefix = name + "_llcode_", suffix = '.s')
    subprocess.check_call([compiler,  '-c', '-emit-llvm', '-S',  src_filename, '-o', assembly_filename, ])
    llvm_source = open(assembly_filename).read()
    print llvm_source
    module = Module.from_assembly(llvm_source)
    
  else: 
    bitcode_filename = tempfile.mktemp(prefix = name + "_bitcode_", suffix = '.o')
    subprocess.check_call([compiler,  '-c', '-emit-llvm',  src_filename, '-o', bitcode_filename, ])
    module = Module.from_bitcode(open(bitcode_filename))
  return module 

def from_c(name, src, compiler = "clang", print_llvm = False):
  module = module_from_c(src, name, compiler, print_llvm)
  shared_module.link_in(module)
  return shared_module.get_function_named(name)



_opt_passes = [
    'targetlibinfo',
    'no-aa',
    'basicaa',
    'memdep',
    'tbaa',
    'instcombine',
    'simplifycfg',
    'basiccg',
    'inline', 
    'functionattrs',
    'argpromotion',
    'memdep',
    'scalarrepl-ssa',
    'sroa',
    'domtree',
    'early-cse',
    'simplify-libcalls',
    'lazy-value-info',
    'correlated-propagation',
    'simplifycfg',
    'instcombine',
    'reassociate',
    'domtree',
    'mem2reg',
    'scev-aa',
    'loops',
    'loop-simplify',
    'lcssa',
    'loop-rotate',
    'licm',
    'lcssa',
    'loop-unswitch',
    'instcombine',
    'scalar-evolution',
    'loop-simplify',
    'lcssa',
    'indvars',
    'loop-idiom',
    'loop-deletion',
    'loop-unroll',
    'memdep',
    'gvn',
    'memdep',
    'dse',
    'adce',
    'correlated-propagation',
    'jump-threading',
    'simplifycfg',
    'instcombine',
  ]


def optimize(llvm_fn, n_iters = 3):
  #engine_builder = llvm.ee.EngineBuilder.new(llvm_fn.module)
  #engine_builder.force_jit()
  #engine_builder.opt(3)
  tm = llvm.ee.TargetMachine.new(opt=3)
  pm, _ = llvm.passes.build_pass_managers(tm, opt = 3,
                                            loop_vectorize = True, 
                                            mod = llvm_fn.module)
   
  #function_pass_manager = module_pass_manager.fpm
  #for p in _opt_passes:
  #  function_pass_manager.add(p)
  for p in _opt_passes:
    pm.add(p)
    #fpm.add(p)
  for _ in xrange(n_iters):
    pm.run(llvm_fn.module)



def get_fn_ptr(llvm_fn, ee = shared_exec_engine):
  FN_PTR_TYPE = lltype_to_ctype(llvm_fn.type.pointee)
  fn_addr = ee.get_pointer_to_function(llvm_fn)
  return FN_PTR_TYPE(fn_addr) 
  
  

def mk_wrapper(fn, step_sizes):
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
  return wrapper 
  