import math
import multiprocessing
import Queue 
import threading

from llvm import *
from llvm.core import * 
import llvm.passes 

from llvm_helpers import * 
import llvm_helpers
import value_helpers 


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

def smallest_divisor(n):
  for i in xrange(2,n):
    if n % i == 0:
      return i 
  return n 


def split_iters(iter_ranges, n_threads = None):
  """
  For now, we just pick the biggest dimension and split 
  it into min(dimsize, 10*n_cores pieces)
  """
  if n_threads is None:
    n_threads = cpu_count()
  
  counts = [safediv(r[1] - r[0], r[2]) for r in iter_ranges]

  biggest_dim = np.argmax(counts)
  dimsize = counts[biggest_dim]
  n_pieces = min(10*n_threads, dimsize)
  factor = float(dimsize) / n_pieces
  pieces = []
  r = iter_ranges[biggest_dim]
  total_start = r[0]

  total_step = r[2]
  for i in xrange(n_pieces):
    # copy all the var ranges, after which we'll modifying 
    # the biggest dimension 
    piece = [r for r in iter_ranges]
    start = total_start + int(math.floor(total_step * factor * i))
    stop = total_start + int(math.floor(total_step * factor * (i+1))) 
    piece[biggest_dim] = (start,stop,total_step)
    pieces.append(piece)
  return pieces   
  
          


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

  return wrapper 


def run(fn, *input_values, **kwds):
  """
  Given a compiled LLVM function and Python input values, 
  convert the input to LLVM generic values and run the 
  function 
  """
  if isinstance(fn, llvm.core.Function):
    ee = kwds.get('ee', shared_exec_engine)
    fn_ptr = llvm_helpers.get_fn_ptr(fn, ee)
  else:
    assert isinstance(fn, ctypes.CFunctionType)
    fn_ptr = fn
  ctypes_inputs = value_helpers.ctypes_values_from_python(input_values)
  return fn_ptr(*ctypes_inputs)

class Worker(threading.Thread):
  def __init__(self, q, work_fn_ptr, fixed_args):
    self.q = q 
    self.ee = ee 
    self.work_fn_ptr = work_fn_ptr

    self.fixed_args = list(fixed_args)
    threading.Thread.__init__(self)

  
  
  def run(self):
    while True:
      try:
        ranges = self.q.get(False)
        # TODO: have to make these types actually match the expected input size
        #starts = [from_python(r[0]) for r in ranges]
        #stops = [from_python(r[1]) for r in ranges]
        starts = [r[0] for r in ranges]
        stops = [r[1] for r in ranges]
        args = self.fixed_args + starts + stops
        self.work_fn_ptr(*args)
        self.q.task_done()
      except Queue.Empty:
        return

        
def parfor(fn, niters, fixed_args = (), ee = None, _cache = {}):
  assert isinstance(fn, Function), \
    "Can only run LLVM functions, not %s" % type(fn)
  assert return_type(fn) == ty_void, \
    "Body of parfor loop must return void, not %s" % return_type(fn)
  
  # in case fixed arguments aren't yet GenericValues, convert them
  #fixed_args = tuple(gv_from_python(v, arg.type) 
  #                   for (v,arg) in 
  #                   zip(fixed_args, fn.args))
  iter_ranges = parse_iters(niters)
  n_fixed = len(fixed_args)
  n_indices = len(iter_ranges)
  cache_key = id(fn) 
  if cache_key in _cache:
    work_fn = _cache[cache_key] 

  else: 
    n_args = n_fixed + n_indices 
    assert len(fn.args) == n_args
    steps = [iter_range[2] for iter_range in iter_ranges]
    work_fn = mk_wrapper(fn, steps)
    if ee is None:
      ee = llvm.ee.ExecutionEngine.new(fn.module)
    # optimize(work_fn, ee)
    _cache[cache_key] = work_fn
   
  q = Queue.Queue()
  # put all the index ranges into the queue
  for work_item in split_iters(iter_ranges):
    q.put(work_item)
  
  fn_ptr = llvm_helpers.get_fn_ptr(work_fn, ee) 
  
  # start worker threads
  for _ in xrange(cpu_count()):
    Worker(q, fn_ptr, fixed_args).start()
  q.join()
  
  
    
  
