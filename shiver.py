import math
import multiprocessing
import Queue 
import threading

from llvm import *
from llvm.core import * 

from llvm_helpers import * 


def safediv(x,y):
  return int(math.ceil(x / float(y)))

def cpu_count():
  return multiprocessing.cpu_count()

def parse_iter_range(x):
  start = None 
  stop = None 
  step = None
  if isinstance(x, int):
    stop = x
  elif isinstance(x, (list, tuple)):
    if len(x) == 1:
      stop = x[0]
    elif len(x) == 2:
      start = x[0]
      stop = x[1]
    else:
      assert len(x) == 3
      start = x[0]
      stop = x[1]
      step = x[2]
  else:
    assert isinstance(x, slice)
    start = x.start 
    stop = x.stop 
    step = x.step 
  
  if start is None:
    start = 0
  if step is None:
    step = 1
  assert isinstance(start, int)
  assert isinstance(stop, int)
  assert isinstance(step, int)
  assert stop > start 
  return (start, stop, step) 


def parse_iters(niters):
  if isinstance(niters, int):
    niters = (niters,)
  assert isinstance(niters, (list, tuple))
  return [parse_iter_range(x) for x in niters]

def split_iters(iter_ranges, n_threads = None):
  if n_threads is None:
    n_threads = cpu_count()
  
  counts = [safediv(r[1] - r[0], r[2]) for r in iter_ranges]
  # total number of iterations across all variables
  total = reduce(lambda x, y: x*y, counts)
  n_pieces = min(10 * n_threads, total)
  assert False
          


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
  exit_builder.ret_void()
  print wrapper
  return wrapper 

class Worker(threading.Thread):
  def __init__(self, q, ee, work_fn, fixed_args):
    self.q = q 
    self.ee = ee 
    self.work_fn = work_fn
    self.fixed_args = list(fixed_args)
    threading.Thread.__init__(self)
  
  def gv_int(self, x):
    return self.ee.GenericValue.int(ty_int64, x) 
  
  def run(self):
    while True:
      try:
        ranges = self.q.get(False)
        
        starts = [self.gv_int(r[0]) for r in ranges]
        stops = [self.gv_int(r[1]) for r in ranges]
        self.ee.run_function(self.work_fn, self.fixed_args + starts + stops)
        self.q.task_done()
      except Queue.Empty:
        return


        
def parfor(fn, niters, fixed_args = (), ee = None):
  assert isinstance(fn, Function), \
    "Can only run LLVM functions, not %s" % type(fn)
  assert return_type(fn) == ty_void, \
    "Body of parfor loop must return void, not %s" % return_type(fn) 
  iter_ranges = parse_iters(niters)
  n_fixed = len(fixed_args)
  n_indices = len(iter_ranges)
  n_args = n_fixed + n_indices 
  assert len(fn.args) == n_args
  steps = [iter_range[2] for iter_range in iter_ranges]
  work_fn = mk_wrapper(fn, steps) 
  if ee is None:
    ee = llvm.ee.ExecutionEngine.new(fn.module)
  q = Queue.Queue()
  # put all the index ranges into the queue
  for work_item in split_iters(iter_ranges):
    q.put(work_item)
  # start worker threads
  for _ in xrange(cpu_count()):
    Worker(q, ee, work_fn, fixed_args).start()
  q.join()
    
  
