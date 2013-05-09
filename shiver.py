import math
import multiprocessing
import numpy as np 
import Queue 
import sys
import threading

from llvm.core import Function 
from llvm.ee import GenericValue 
from llvm_helpers import mk_wrapper, return_type, shared_exec_engine, optimize
from llvm_helpers import module_from_c, from_c  

from type_helpers import is_llvm_float_type, is_llvm_int_type, is_llvm_ptr_type
from type_helpers import lltype_to_dtype, ty_int8, ty_int64, ty_float64, ty_void  
from type_helpers import dtype_to_ctype_name 

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
  largest_dim = np.max(counts)
  total_count = float(np.sum(counts))
  split_factors = [ (c / total_count) ** 3 for c in counts ]
  n_pieces = min(2*n_threads, largest_dim)

  
  # initialize work_items with an empty single range 
  work_items = [[]]
  for (dim_idx,dim_count) in enumerate(counts):

    dim_start, _, dim_step = iter_ranges[dim_idx]
    n_dim_pieces = int(math.ceil(split_factors[dim_idx] * n_pieces))
    dim_factor = float(dim_count) / n_dim_pieces
    
    old_work_items = [p for p in work_items]
    work_items = []
    for i in xrange(n_dim_pieces):
      # copy all the var ranges, after which we'll modifying 
      # the biggest dimension 

      start = dim_start + int(math.floor(dim_step * dim_factor * i))
      stop = dim_start + int(math.floor(dim_step * dim_factor * (i+1)))
      
      dim_work_item = (start,stop,dim_step)
      for old_work_item in old_work_items:
        new_work_item = [r for r in old_work_item]
        new_work_item.append(dim_work_item) 
        work_items.append(new_work_item)
 
  return work_items 
  
          

def gv_from_python(x, llvm_type = None):
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
    dtype = lltype_to_dtype(elt_type) 
    assert dtype == x.dtype, \
        "Can't pass array with %s* data pointer to function that expects %s*" % (x.dtype, dtype)
    return GenericValue.pointer(x.ctypes.data)
  

def run_with_generic_values(fn, gv_inputs, ee):  
  n_given = len(gv_inputs)
  n_expected = len(gv_inputs)
  assert n_given == n_expected, "Expected %d inputs but got %d" % (n_expected, n_given)
  return ee.run_function(fn, gv_inputs)

def run(fn, *input_values, **kwds):
  """
  Given a compiled LLVM function and Python input values, 
  convert the input to LLVM generic values and run the 
  function 
  """
  
  ee = kwds.get('ee', shared_exec_engine)
  input_types = [arg.type for arg in fn.args]
  gv_inputs = [gv_from_python(x, t) 
               for (x,t) in 
               zip(input_values, input_types)]
  
  return run_with_generic_values(fn, gv_inputs, ee)


class Worker(threading.Thread):
  def __init__(self, q, ee, llvm_fn, fixed_args):
    self.ee = ee 
    self.q = q 
    self.llvm_fn = llvm_fn
    self.fixed_args = list(fixed_args)    
    threading.Thread.__init__(self)
  
  
  def run(self):
    while True:
      try:
        ranges = self.q.get(False)
        # TODO: have to make these types actually match the expected input size
        index_types = [arg.type for arg in self.llvm_fn.args[-len(ranges):]]
        starts = [gv_from_python(r[0], t) 
                  for (r,t) in 
                  zip(ranges, index_types)]
        stops = [gv_from_python(r[1], t) 
                  for (r,t) in 
                  zip(ranges, index_types)]        
        args = self.fixed_args + starts + stops
        run_with_generic_values(self.llvm_fn, args, self.ee)
        self.q.task_done()
      except Queue.Empty:
        return

        
def parfor(fn, niters, fixed_args = (), ee = shared_exec_engine, _cache = {}):
  assert isinstance(fn, Function), \
    "Can only run LLVM functions, not %s" % type(fn)
  assert return_type(fn) == ty_void, \
    "Body of parfor loop must return void, not %s" % return_type(fn)
  
  # in case fixed arguments aren't yet GenericValues, convert them
  fixed_args = tuple(gv_from_python(v, arg.type) 
                     for (v,arg) in 
                     zip(fixed_args, fn.args))
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

    optimize(work_fn)
 
    _cache[cache_key] = work_fn
   
  q = Queue.Queue()
  # put all the index ranges into the queue
  for work_item in split_iters(iter_ranges):
    q.put(work_item)
  
  
  # start worker threads
  for _ in xrange(cpu_count()):
    Worker(q, ee, work_fn, fixed_args).start()
  q.join()
  
  
    
  
