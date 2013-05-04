import math
import multiprocessing
import threading

import llvm 
import llvm.core 

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
    
# TODO: Make threads that call ee.run_function with fixed_args + (GenericValue.int(start), etc...) 
def run(fn, niters, fixed_args = (), ee = None):
  assert isinstance(fn, llvm.core.Function)
  slices = parse_iters(niters)
  assert len(fn.args) == len(fixed_args) + len(slices)  
  split_slices = split_iters(slices)   
  if ee is None:
    ee = llvm.ee.ExecutionEngine.new(fn.module)
  
     
    
  
