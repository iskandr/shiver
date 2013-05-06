import numpy as np


from llvm import * 
from llvm.core import * 

ty_int8 = Type.int(8)
ty_int16 = Type.int(16) 
ty_int32 = Type.int(32) 
ty_int64 = Type.int(64) 

ty_float32 = Type.float()
ty_float64 = Type.double()

ty_ptr_int8 = Type.pointer(ty_int8)
ty_ptr_int16 = Type.pointer(ty_int16)
ty_ptr_int32 = Type.pointer(ty_int32)
ty_ptr_int64 = Type.pointer(ty_int64)

ty_ptr_float32 = Type.pointer(ty_float32)
ty_ptr_float64 = Type.double(ty_float64)

mappings = {
  np.int8 : ty_int8,             
  np.int16 : ty_int16, 
  np.int32 : ty_int32, 
  np.int64 : ty_int64, 
  np.float32 : ty_float32, 
  np.float64 : ty_float64,
  int : ty_int64, 
  float : ty_float64, 
  bool : ty_int8, 
}

def ptr(t):
  return Type.pointer(t)

def to_lltype(t):
  """
  Convert python types to LLVM types.
  Examples: 
    to_lltype(int) == Type.int(64)
    to_lltype([float]) == Type.pointer(Type.double())
  """
  if isinstance(t, np.dtype):
    return to_lltype(t.type)
  elif isinstance(t, (list,tuple)):
    assert len(t) == 1
    elt_t = to_lltype(t[0])
    return ptr(elt_t) 
  assert t in mappings, "Unknown type %s" % (t,)
  return mappings[t]