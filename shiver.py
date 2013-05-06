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
        
        starts = [self.gv_int(r.start) for r in ranges]
        stops = [self.gv_int(r.stop) for r in ranges]
        self.ee.run_function(self.work_fn, self.fixed_args + starts + stops)
        self.q.task_done()
      except Queue.Empty:
        return
        
def parfor(fn, niters, fixed_args = (), ee = None):
  assert isinstance(fn, Function), \
    "Can only run LLVM functions, not %s" % type(fn)
  assert return_type(fn) == ty_void, \
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
    
  
