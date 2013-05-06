Shiver
======
*a shiver of sharks, biting and tearing, until all is consumed*

A multi-threaded work queue for functions compiled with [llvmpy](http://http://www.llvmpy.org/). 
Give *Shiver* a function whose last argument is an index (or multiple indices) and a description of the 
iteration space (i.e. a integer number of iterations, multiple integers, or even slice objects with start/stop/step fields), 
and shiver does all the messy plumbing of running your code in parallel. 

Example:

```python
   # call a simple function that takes an index as its only input
   shiver.parfor(fn_simple, niters=10)
   
   # call a function which takes two integers with all pairs 
   # of even integers [2..10] and [10..40]
   shiver.parfor(fn_two_inputs, niters = (slice(2,11,2), slice(10,40,2)))

   # more complex function which takes first two fixed array arguments and then
   # a varying integer index which will take values 30,33,36,39, etc... 
   x_gv = GenericValue.pointer(x.ctypes.data)
   y_gv = GenericValue.pointer(y.ctypes.data)
   shiver.parfor(fn_three_inputs, fixed_args=[x_gv, y_gv], niters=slice(30,140,3))
```


FAQ
----

*Can I use this library to run Python code in parallel?* 
    
    Sorry, no. Shiver is only useful if you're already compiling [LLVM](http://www.drdobbs.com/architecture-and-design/the-design-of-llvm/240001128) code using [llvmpy](http://www.llvmpy.org/). Shiver takes a compiled function which constitutes the body of a loop (or a nesting of loops) and runs that code in parallel. It uses the Python threading API to split up your work and saves from having to deal with [pthreads](http://www.cs.fsu.edu/~baker/realtime/restricted/notes/pthreads.html). 


*You're using Python threads, doesn't that mean you're still stuck behind the [GIL](http://stackoverflow.com/questions/1294382/what-is-a-global-interpreter-lock-gil)?*

     When Shiver calls into native code (using [ExecutionEngine.run_function](http://sourcecodebrowser.com/llvm-py/0.5plus-psvn85/classllvm_1_1ee_1_1_execution_engine.html#a4da1e185faa9926638751f2bde570ad2)), it releases the Global Interpreter Lock, allowing its threads to actually utilize all of your processors. 
