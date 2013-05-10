Shiver
======
*a shiver of sharks, biting and tearing, until all is consumed*

A multi-threaded work queue for functions compiled with [llvmpy](http://http://www.llvmpy.org/). 
Give *Shiver* a function whose last argument is an index (or multiple indices) and an iteration space
 (i.e. a number of iterations, a tuple of integers, or even slice objects with start/stop/step fields), 
and shiver does all the messy plumbing of running your code in parallel. 

Example:

```python

   # we're going to fill this array with the numbers [0...9]
   x = np.empty(10, dtype=int)

   # compile an LLVM function which takes an array, and an index
   fn1 = shiver.from_c("fn1", "void fn1(long *x, long i) { x[i] = i;}")   

   # run fn_one_idx in parallel;
   # - shiver will supply x's data pointer as a fixed argument to all threads 
   # - each worker thread will also get a subrange of the indices [0..9]
   # - the numpy array 'x' will be passed in as the underlying pointer x.ctypes.data 
   shiver.parfor(fn1, niters=len(x), fixed_args = [x])

   # if the function you compile returns a value, 
   # then shiver will collect those values into a result array 
   ident = shiver.from_c("indentity", "long identity(long i) { return i; }")
   y = shiver.parfor(ident, 10)
   assert (y==x).all()
    
   # let's do the same thing again, but here we'll explicitly convert the 
   # fixed array argument to a type LLVM understands 
   x_gv = GenericValue.pointer(x.ctypes.data)
   shiver.parfor(fn1, niters=len(x), fixed_args = [x_gv])
   
   # Now we'll build a function which takes two indices which range 
   # over all pairs of integers [0..9] and [0..20] and fills x with their products
   src = "float mult(long i, long j) { return (float) i*j; }" 
   fn2 = shiver.from_c("mult", src)
   result_grid = shiver.parfor(fn2, (10,20)
   assert result_grid.shape == (10,20)

```


FAQ
----

*Can I use this library to run Python code in parallel?* 
    
Sorry, no. Shiver is only useful if you're already compiling [LLVM](http://www.drdobbs.com/architecture-and-design/the-design-of-llvm/240001128) code using [llvmpy](http://www.llvmpy.org/) or if you want to use shiver's *from_c* helper to compile simple C functions. Shiver takes an LLVM function which constitutes the body of a loop (or a nesting of loops) and runs that code in parallel. It uses the Python threading API to split up your work and saves you from having to deal with [pthreads](http://www.cs.fsu.edu/~baker/realtime/restricted/notes/pthreads.html). 


*You're using Python threads, doesn't that mean you're still stuck behind the [GIL](http://stackoverflow.com/questions/1294382/what-is-a-global-interpreter-lock-gil)?*

When Shiver calls into native code it releases the Global Interpreter Lock, allowing its threads to actually utilize all of your processors. 
