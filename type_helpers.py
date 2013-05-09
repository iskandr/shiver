import numpy as np 


import llvm.core as llc
from llvm.core import Type as lltype  

ty_void = lltype.void()
ty_int8 = lltype.int(8)
ty_int16 = lltype.int(16) 
ty_int32 = lltype.int(32) 
ty_int64 = lltype.int(64) 

ty_float32 = lltype.float()
ty_float64 = lltype.double()

ty_pyobj = lltype.opaque("PyObj")
ty_ptr_pyobj = lltype.pointer(ty_pyobj)

ty_ptr_int8 = lltype.pointer(ty_int8)
ty_ptr_int16 = lltype.pointer(ty_int16)
ty_ptr_int32 = lltype.pointer(ty_int32)
ty_ptr_int64 = lltype.pointer(ty_int64)

ty_ptr_float32 = lltype.pointer(ty_float32)
ty_ptr_float64 = lltype.pointer(ty_float64)

python_to_lltype_mappings = {
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

def python_to_lltype(t):
  """
  Convert python types to LLVM types.
  As a convenience, you can specify lists of types to indicate
  arrays of that scalar type. 
  Examples: 
    to_lltype(int) == Type.int(64)
    to_lltype([float]) == Type.pointer(Type.double())
  """
  if isinstance(t, np.dtype):
    return python_to_lltype(t.type)
  elif isinstance(t, (list,tuple)):
    assert len(t) == 1
    elt_t = python_to_lltype(t[0])
    return lltype.pointer(elt_t) 
  assert t in python_to_lltype_mappings, "Unknown type %s" % (t,)
  return python_to_lltype_mappings[t]

lltype_name_to_numpy_mappings = {
  str(ty_int8) : np.int8,              
  str(ty_int16) : np.int16, 
  str(ty_int32) : np.int32,  
  str(ty_int64) : np.int64,  
  str(ty_float32) : np.float32,  
  str(ty_float64) : np.float64,
}

def lltype_to_numpy_type(t):
  """
  Converts an LLVM scalar type to the equivalent NumPy type
  """
  t_str = str(t)
  assert t_str in lltype_name_to_numpy_mappings, \
      "Don't know how to convert LLVM type %s to numpy type" % (t_str,)
  return lltype_name_to_numpy_mappings[t_str]
  
lltype_name_to_dtype_mappings = {}
for (k,v) in lltype_name_to_numpy_mappings.iteritems():
  lltype_name_to_dtype_mappings[k] = np.dtype(v)

def lltype_to_dtype(t):
  """
  Converts an LLVM scalar type to the equivalent NumPy dtype
  """
  t_str = str(t)
  assert t_str in lltype_name_to_dtype_mappings, \
      "Don't know how to convert LLVM type %s to dtype" % (t_str,)
  return lltype_name_to_dtype_mappings[t_str]

def is_llvm_float_type(t):
  return t.kind in (llc.TYPE_FLOAT, llc.TYPE_DOUBLE)

def is_llvm_int_type(t):
  return t.kind == llc.TYPE_INTEGER

def is_llvm_ptr_type(t):
  return t.kind == llc.TYPE_POINTER

from ctypes import c_int8, c_int16, c_int32, c_int64
from ctypes import c_float, c_double
from ctypes import CFUNCTYPE, POINTER

c_int8_p = POINTER(c_int8)
c_int16_p = POINTER(c_int16)
c_int32_p = POINTER(c_int32)
c_int64_p = POINTER(c_int64)
c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)

llvm_to_ctypes_mapping = {
  str(ty_void) : None, 
  str(ty_int8) : c_int8, 
  str(ty_int16) : c_int16, 
  str(ty_int32) : c_int32, 
  str(ty_int64) : c_int64, 
  str(ty_float32) : c_float, 
  str(ty_float64) : c_double,
  str(ty_ptr_int8) : POINTER(c_int8),  
  str(ty_ptr_int16) : POINTER(c_int16), 
  str(ty_ptr_int32) : POINTER(c_int32), 
  str(ty_ptr_int64) : POINTER(c_int64),                                    
  str(ty_ptr_float32) : POINTER(c_float),
  str(ty_ptr_float64) : POINTER(c_double),
}

def lltype_to_ctype(lltype):
  """
  Convert an LLVM type into its corresponding ctypes type.
  For now only works for scalars, data pointers, and direct 
  function types (but not function pointers). 
  """
  if isinstance(lltype, llc.FunctionType):
    return_type = lltype_to_ctype(lltype.return_type)
    arg_types = [lltype_to_ctype(arg_t) for arg_t in lltype.args]
    ctypes_fn_t = CFUNCTYPE(return_type, *arg_types)
    # should I add ctypes_fn_t to the mapping?
    return ctypes_fn_t 
    

  t_str = str(lltype)
  assert t_str in llvm_to_ctypes_mapping, \
      "Don't know how to convert LLVM type %s to ctypes" % (t_str,)
  return llvm_to_ctypes_mapping[t_str] 


def is_ctypes_real_type(t):
  return t in (c_float, c_double)

def is_ctypes_int_type(t):
  return t in (c_int8, c_int16, c_int32, c_int64)

def is_ctypes_ptr_type(t):
  return hasattr(t, 'contents')

def python_to_ctype(t):
  lltype = python_to_lltype(t)
  return lltype_to_ctype(lltype)

def dtype_to_ctype(t):
  assert isinstance(t, np.dtype)
  return python_to_ctype(t.type)

_ctypes_names = { 
  c_int8 : 'char', 
  c_int16 : 'int16_t', 
  c_int32 : 'int', 
  c_int64 : 'long', 
  c_float : 'float', 
  c_double :'double',                  
}

def ctype_name(ct):
  return _ctypes_names[ct]

def dtype_to_ctype_name(t):
  return ctype_name(dtype_to_ctype(t))


ctype_to_lltype_mapping = {
  c_int8 : ty_int8,   
  c_int16 : ty_int16,  
  c_int32 : ty_int32,  
  c_int64 : ty_int64,  
  c_float : ty_float32,  
  c_double : ty_float64,  
  c_int8_p : ty_ptr_int8,    
  c_int16_p : ty_ptr_int16,  
  c_int32_p : ty_ptr_int32,  
  c_int64_p : ty_ptr_int64,                                     
  c_float_p : ty_ptr_float32, 
  c_double_p : ty_ptr_float64
}

def ctype_to_lltype(ctype):
  if ctype is None:
    return ty_void
  if hasattr(ctype, 'argtypes'):
    llvm_argtypes = [ctype_to_lltype(arg_t) for arg_t in ctype.argtypes]
    llvm_ret_type = ctype_to_lltype(ctype.restype)
    return lltype.function(llvm_ret_type, llvm_argtypes) 
  else: 
    assert ctype in ctype_to_lltype_mapping, \
      "Don't know how to convert C type %s to LLVM type" % (ctype,)
    return llvm_to_ctypes_mapping[ctype] 

def ctype_to_dtype(t):
  lltype = ctype_to_lltype(t)
  return lltype_to_dtype(lltype)