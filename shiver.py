import threading
import llvm 
import llvm.core 

def parse_iter(x):
  if isinstance(x, int):
    s = slice(x)
  elif isinstance(x, tuple):
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
  return s 

def parse_iters(niters):
  if isinstance(niters, int):
    niters = (niters,)
  assert isinstance(niters, tuple)
  nvars = len(niters)
  slices = []
  totals = []
  for x in niters:
    s = parse_iter(x)
    total = safediv(s.stop - s.start, s.step)
    assert total > 0
    slices.append(s)
    totals.append(s)
  return slices, totals
    
def run(fn, niters, fixed_args = ()):
  assert isinstance(fn, llvm.core.Function)
  slices, totals = parse_iters(niters)
  
     
    
  
