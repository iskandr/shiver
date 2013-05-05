import math
import multiprocessing
import threading

Module.new('my_module')

from llvm import *
from llvm.core import * 


def savediv(x,y):
  return int(math.ceil(x / float(y)))

def cpu_count():
  return multiprocessing.cpu_count()

def parse_iter(x):
  if isinstance(x, int):
    s = slice(x)
  elif isinstance(x, (list, tuple)):
    if len(x) == 1:
      s = slice(x[0])
    elif len(x) == 2:
      s = slice(x[0], x[1])
    else:
      assert len(x) == 3
      s = slice(x[0], x[1], x[2])
  else:
    assert isinstance(x, slice)
    s = x
  if s.start is None:
    s.start = 0
  if s.step is None:
    s.step = 1
  assert isinstance(s.start, int)
  assert isinstance(s.stop, int)
  assert isinstance(s.step, int)
  assert s.stop > s.start 
  return s 


def parse_iters(niters):
  if isinstance(niters, int):
    niters = (niters,)
  assert isinstance(niters, (list, tuple))
  return [parse_iter(x) for x in niters]

def split_iters(slices, n_pieces):
  counts = [safediv(s.stop - s.start, s.step) for s in slices]
  # total number of iterations across all variables
  total = reduce(lambda x, y: x*y, counts)
  n_pieces = min(10 * cpu_count(), total)

def return_type(fn):
  return fn.type.pointee.return_type 
    
def mk_wrapper(fn, step_sizes):
  n_indices = len(step_sizes)
  m = fn.module
  ty_int = Type.int(64) 
  # need to double the integer index inputs to allow for a stop argument 
  # of each   the number of index inputs to
  old_input_types = [arg.type for arg in f.args]
  old_input_names = [arg.name for arg in f.args]
  extra_index_input_types = [ty_int for _ in xrange(n_indices)]
  extra_index_names = ["stop%d" % (i+1) for i in xrange(n_indices)]
  new_input_types = old_input_types + extra_input_types
  ty_func = Type.function(return_type(fn), new_input_types)
  new_name = f.name + "_wrapper" 
  wrapper = fn.module.add_function(ty_func, new_name)
  for arg_idx in xrange(len(new_input_types)):
    if arg_idx < len(old_input_types):
      wrapper.args[arg_idx].name = fn.args[arg_idx].name 
    else:
      wrapper.args[arg_idx].name = fn.args[arg_idx - n_indices].name
  # TODO: create builder, make n_indices loops in the body
  bb = f_sum.append_basic_block("entry")
  builder = Builder.new(bb)
  
  return wrapper 

 
def parfor(fn, niters, fixed_args = (), ee = None):
  assert isinstance(fn, Function), \
    "Can only run LLVM functions, not %s" % type(fn)
  assert return_type(fn) == Type.void, \
    "Body of parfor loop must return void, not %s" % return_type(fn) 
  slices = parse_iters(niters)
  n_fixed = len(fixed_args)
  n_indices = len(slices)
  n_args = n_fixed + n_indices 
  assert len(fn.args) == n_args
  split_slices = split_iters(slices)   
  if ee is None:
    ee = llvm.ee.ExecutionEngine.new(fn.module)
  work_fn = mk_wrapper(fn, n_indices) 
   
    
  
