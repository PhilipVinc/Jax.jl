# Jax.jl
An experiment carried out over a lazy confined sunday.

This wraps some functionality of Jax in julia, attempting to make Jax able to trace through native Julia functions, and compute their gradients. The package also attempts to play nicely with Flux.

To install it you must have jax and jaxlib installed throuh pip/conda in PyCall's python environment.
```julia
pkg"add https://github.com/PhilipVinc/Jax.jl
using Conda
Conda.add("jax")
```

The main type defined by this package is `JaxArray` which wraps a python object. You can cast any dense array to `JaxArray`, and if you have a Flux model you can use the `|> tojax` function much like you'd use `|> gpu`.

Code that normally works on Julia Arrays/CuArrays should work out of the box with JaxArrays and (hopefully) yield the same results.

An important note is that, since Jax does not support Fortran memory ordering (like julia), all arrays are transposed when passed to Jax, to allow to perform operations efficiently. Likewise, all (wrapped) reduction operations will transpose the axis of the reduction. This should be transparent when you use it.

Julia's broadcasting is overloaded, in order to call the correct Jax (python) operations. In order for this to work, if you define some functions that you want to broadcast you must redefine them with the `@jaxfunc` macro, similarly to what you would do with `CuArrays`.

```
julia> using Jax
julia> key = Jax.Random.PRNGKey(0)
Jax PRNG Key UInt32[0x00000000, 0x00000000]

julia> A = rand(key, 3,4)
4×3 JaxArray{Float32,2}:
 0.883009  0.347568   0.415125
 0.135734  0.755726   0.161128
 0.671362  0.677308   0.248591
 0.725237  0.0192928  0.607814

julia> f(x) = x * x + x ^ 2
julia> f.(A)
ERROR: PyError ($(Expr(:escape, :(ccall(#= /Users/filippovicentini/.julia/packages/PyCall/zqDXB/src/pyfncall.jl:43 =# @pysym(:PyObject_Call), PyPtr, (PyPtr, PyPtr, PyPtr), o, pyargsptr, kw))))) <class 'TypeError'>

julia> Jax.@jaxfunc f(x) = x * x + x ^ 2
julia> f.(A)
4×3 JaxArray{Float32,2}:
 1.55941    0.241607     0.344657
 0.0368474  1.14224      0.0519247
 0.901455   0.917491     0.123595
 1.05194    0.000744427  0.738875

```

Of course the performance of all this will be quite low because the code will keep jumping between Python and Julia. However, you (again, hopefully) should be able to jit the resulting code, which will make it so that it will jump all those hoops only once, and the subsequent times it will run the compiled operations.

Also vmap is supported.
Check `jax.jit` and `jax.vmap`.
