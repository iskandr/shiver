import math
import multiprocessing
import Queue 
import threading

from llvm import *
from llvm.core import * 

ty_int = Type.int(64) 

def safediv(x,y):
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


class LoopBuilder(object):
  def __init__(self, original_fn, closure_values, 
               start_values, stop_values, step_constants):
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
    
  def create(self, bb, builder, loop_vars = []):
    n = len(loop_vars)
    if n == self.n_loops:
      builder.call(self.original_fn, self.closure_values + loop_vars)
      return bb 
    else:
      next_bb = None
      idx_var = None
      self.create(next_bb, builder, loop_vars + [idx_var] )
  
          

def mk_wrapper(fn, step_sizes):
  n_indices = len(step_sizes)

  # need to double the integer index inputs to allow for a stop argument 
  # of each   the number of index inputs to
  old_input_types = [arg.type for arg in fn.args]
  extra_input_types = [ty_int for _ in xrange(n_indices)]
  new_input_types = old_input_types + extra_input_types
  ty_func = Type.function(return_type(fn), new_input_types)
  new_name = fn.name + "_wrapper" 
  wrapper = fn.module.add_function(ty_func, new_name)
  for arg_idx in xrange(len(new_input_types)):
    if arg_idx < len(old_input_types):
      wrapper.args[arg_idx].name = fn.args[arg_idx].name 
    else:
      wrapper.args[arg_idx].name = fn.args[arg_idx - n_indices].name + "_stop"
  # TODO: create builder, make n_indices loops in the body
  index_vars = wrapper.args[-2*n_indices:]
  closure_vars = wrapper.args[:-2*n_indices]
  start_vars = index_vars[:n_indices]
  stop_vars = index_vars[n_indices:]
  
  assert len(start_vars) == len(stop_vars)
  loop_builder = LoopBuilder(fn, 
                             closure_vars, 
                             start_vars, 
                             stop_vars, 
                             step_sizes)
  entry_bb = wrapper.append_basic_block("entry")
  builder = Builder.new(entry_bb)
  exit_bb, exit_builder = loop_builder.create(entry_bb, builder)
  exit_builder.ret()
  # how to tie the exit bb in to the loop?
  return wrapper 

class Worker(threading.Thread):
  def __init__(self, q, ee, work_fn, fixed_args):
    self.q = q 
    self.ee = ee 
    self.work_fn = work_fn
    self.fixed_args = list(fixed_args)
    threading.Thread.__init__(self)
  
  def gv_int(self, x):
    return self.ee.GenericValue.int(ty_int, x) 
  
  def run(self):
    while True:
      try:
        ranges = self.q.get(False)
        
        starts = [self.gv_int(r.start) for r in ranges]
        stops = [self.gv_int(r.stop) for r in ranges]
        self.ee.run_function(self.work_fn, self.fixed_args + starts + stops)
        self.q.task_done()
      except Queue.Empty:
        return
        
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
  work_fn = mk_wrapper(fn, n_indices) 
  if ee is None:
    ee = llvm.ee.ExecutionEngine.new(fn.module)
  q = Queue.Queue()
  # put all the index ranges into the queue
  for work_item in split_iters(slices):
    q.put(work_item)
  # start worker threads
  for _ in xrange(cpu_count()):
    Worker(q, ee, work_fn, fixed_args).start()
  q.join()
    
  
