import numpy as np 
from llvm.ee import GenericValue
from type_helpers import ty_int64, ty_float64, ty_int8
from type_helpers import is_llvm_float_type, is_llvm_int_type
from type_helpers import is_llvm_ptr_type, lltype_to_dtype

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
  



def ctypes_value_from_python(x, ctype = None):
  if isinstance(x, (int,long)):
    llvm_type =  ty_int64 if ctype is None else 
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
    elt_type_str = str(elt_type) 
    assert elt_type_str in to_dtype_mappings, \
      "Don't know how to convert LLVM type %s to dtype" % (elt_type_str,)
    dtype = to_dtype_mappings[elt_type_str]
    assert dtype == x.dtype, \
        "Can't pass array with %s* data pointer to function that expects %s*" % (x.dtype, dtype)