shiver
======

A multi-threaded work queue for functions compiled with [llvmpy](http://http://www.llvmpy.org/). 
Give *shiver* a function whose last argument is an index (or multiple indices) and a description of the 
iteration space (i.e. a integer number of iterations, multiple integers, or even slice objects with start/stop/step fields), 
and shiver does all the messy plumbing of running your code in parallel. 

Example:

```python
   # call a simple function that takes an index as its only input
   shiver.run(fn_simple, niters=10)
   
   # call a function which takes two integers with all pairs 
   # of even integers [2..10] and [10..40]
   shiver.run(fn_two_inputs, niters = (slice(2,11,2), slice(10,40,2)))

   # more complex function which takes first two fixed array arguments and then
   # a varying integer index which will take values 30,33,36,39, etc... 
   x_gv = GenericValue.pointer(x.ctypes.data)
   y_gv = GenericValue.pointer(y.ctypes.data)
   shiver.run(fn_three_inputs, fixed_args=[x_gv, y_gv), niters=slice(30,140,3))
```
