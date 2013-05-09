

import subprocess 
import tempfile

from llvm import * 
from llvm.core import *
from llvm.ee import GenericValue, ExecutionEngine 
from type_helpers import ty_int64, ty_float64, ty_void, lltype_to_ctype

shared_module = llvm.core.Module.new("shiver_global_module")



  
shared_exec_engine = llvm.ee.ExecutionEngine.new(shared_module)

def return_type(fn):
  return fn.type.pointee.return_type 

def input_types(fn):
  return [arg.type for arg in fn.args]  

def input_names(fn):
  return [arg.name for arg in fn.args]

def const_int(x, t = ty_int64):
  return Constant.int(t, x)

def const_float(x, t = ty_float64):
  return Constant.real(t, x)

def const(x):
  if isinstance(x, int):
    return const_int(x)
  else:
    assert isinstance(x, float)
    return const_float(x)



def empty_fn( name, input_types, output_type = ty_void, module = shared_module):
  names = []
  types = []
  for (i, item) in enumerate(input_types):
    if isinstance(item, (list, tuple)):
      names.append(item[0])
      types.append(item[1])
    else:
      names.append("arg%d" % (i+1))
      types.append(item)
  
  ty_func = Type.function(output_type, types)
  fn = module.add_function(ty_func, name)
  for (i,arg) in enumerate(fn.args):
    arg.name = names[i]
  return fn  


def module_from_c(src, name = 'fn', compiler = 'clang', print_llvm = False):
  src_filename = tempfile.mktemp(prefix = name + "_src_", suffix = '.c')

  f = open(src_filename, 'w')
  f.write(src + '\n')
  f.close()
  if print_llvm:
    assembly_filename = tempfile.mktemp(prefix = name + "_llcode_", suffix = '.s')
    subprocess.check_call([compiler,  '-c', '-emit-llvm', '-S',  src_filename, '-o', assembly_filename, ])
    llvm_source = open(assembly_filename).read()
    print llvm_source
    module = Module.from_assembly(llvm_source)
    
  else: 
    bitcode_filename = tempfile.mktemp(prefix = name + "_bitcode_", suffix = '.o')
    subprocess.check_call([compiler,  '-c', '-emit-llvm',  src_filename, '-o', bitcode_filename, ])
    module = Module.from_bitcode(open(bitcode_filename))
  return module 

_save_modules = []
def from_c(name, src, compiler = "clang", print_llvm = False, link = False):
  module = module_from_c(src, name, compiler, print_llvm)
  if link:
    shared_module.link_in(module)
  else:
    # prevent the module from being deleted 
    # which then invalidates the returned function 
    _save_modules.append(module)
  return module.get_function_named(name)



_opt_passes = [

    'targetlibinfo',
    'no-aa',
    'basicaa',
    'memdep',
    'tbaa',
    'instcombine',
    'simplifycfg',
    'basiccg',
    'verify', 

    'memdep',
    'scalarrepl-ssa',
    'sroa',
    'domtree',
    'early-cse',
    'simplify-libcalls',
    'lazy-value-info',
    'correlated-propagation',
    'simplifycfg',
    'instcombine',
    'reassociate',
    'domtree',
    'mem2reg',
    'scev-aa',
    'loops',
    'loop-simplify',
    'lcssa',
    'loop-rotate',
    'licm',
    'lcssa',
    'loop-unswitch',
    'instcombine',
    'scalar-evolution',
    'loop-simplify',
    'lcssa',
    'indvars',
    'loop-idiom',
    'loop-deletion',
    'loop-unroll',
    'memdep',
    'gvn',
    'memdep',
    'dse',
    'adce',
    'correlated-propagation',
    'jump-threading',
    'simplifycfg',
    'instcombine',
  ]


def optimize(llvm_fn, n_iters = 4, module_opts = False):

  tm = llvm.ee.TargetMachine.new(opt=3)
  pm, fpm = llvm.passes.build_pass_managers(tm, opt = 3,
                                            loop_vectorize = True,  
                                            mod = llvm_fn.module)
  
  
  if module_opts: 
    for p in ['functionattrs', 
              'argpromotion', 
              'inline', 
              'memdep', 
              'loop-unroll']:  
      pm.add(p)
    
    pm.run(llvm_fn.module)
    pm.run(llvm_fn.module)
  
  for p in _opt_passes:
    fpm.add(p)

  for _ in xrange(n_iters):
    fpm.run(llvm_fn)


def get_fn_ptr(llvm_fn, ee = shared_exec_engine):
  FN_PTR_TYPE = lltype_to_ctype(llvm_fn.type.pointee)
  fn_addr = ee.get_pointer_to_function(llvm_fn)
  return FN_PTR_TYPE(fn_addr) 
