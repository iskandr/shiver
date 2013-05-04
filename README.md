shiver
======

A multi-threaded work queue for functions compiled with [llvmpy](http://http://www.llvmpy.org/). 
Give *shiver* a function whose last argument is an index (or multiple indices) and a description of the 
iteration space (i.e. a integer number of iterations, multiple integers, or even slice objects with start/stop/step fields), 
and shiver does all the messy plumbing of running your code in parallel. 

Example:

```python
   # fn takes three integers, the last two are index values
   shiver.run(fn, fixed_args=(GenericValue.int(1)), niters=[10, slice(start=2, stop=12, step = 2)])
```
