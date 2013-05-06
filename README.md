Shiver
======
*a shiver of sharks, biting and tearing, until all is consumed*

A multi-threaded work queue for functions compiled with [llvmpy](http://http://www.llvmpy.org/). 
Give *Shiver* a function whose last argument is an index (or multiple indices) and a description of the 
iteration space (i.e. a integer number of iterations, multiple integers, or even slice objects with start/stop/step fields), 
and shiver does all the messy plumbing of running your code in parallel. 

Example:

```python
   # we're going to fill this array with the numbers [0...9]
   x = np.empty(10, dtype=int)
   # compile an LLVM function which takes a data point, and an index
   fn_one_idx = shiver.from_c("void int(long *x, long i) { x[i] = i;}")   
   # run fn_one_idx in parallel;
   # - shiver will supply x's data pointer as a fixed argument to all threads 
   # - each worker thread will also get a subrange of the indices [0..9]
   shiver.parfor(fn_one_idx, niters=len(x), fixed_args = [x])
   
   # let's do the same thing again, but here we'll explicitly convert the 
   # fixed array argument to a type LLVM understands 
   x_gv = GenericValue.pointer(x.ctypes.data)
   shiver.parfor(fn_one_idx, niters=len(x), fixed_args = [x_gv])
   
   # Now we'll build a function which takes two indices which range 
   # over all pairs of integers [1..5] and [1..2] and fills x with their products
   src = "void int(long* x, long i, long j) { x[(j-1)*5 + i-1] = i*j;}" 
   fn_two_idxs = shiver.from_c(src)

   # notice that we're passing in 
   shiver.parfor(fn_two_idxs, niters = (slice(1,6), slice(1,3)), fixed_args =[x])
```


FAQ
----

*Can I use this library to run Python code in parallel?* 
    
Sorry, no. Shiver is only useful if you're already compiling [LLVM](http://www.drdobbs.com/architecture-and-design/the-design-of-llvm/240001128) code using [llvmpy](http://www.llvmpy.org/). Shiver takes a compiled function which constitutes the body of a loop (or a nesting of loops) and runs that code in parallel. It uses the Python threading API to split up your work and saves from having to deal with [pthreads](http://www.cs.fsu.edu/~baker/realtime/restricted/notes/pthreads.html). 


*You're using Python threads, doesn't that mean you're still stuck behind the [GIL](http://stackoverflow.com/questions/1294382/what-is-a-global-interpreter-lock-gil)?*

When Shiver calls into native code (using [ExecutionEngine.run_function](http://sourcecodebrowser.com/llvm-py/0.5plus-psvn85/classllvm_1_1ee_1_1_execution_engine.html#a4da1e185faa9926638751f2bde570ad2)), it releases the Global Interpreter Lock, allowing its threads to actually utilize all of your processors. 
