"""
    jit(fun, static_argnums=tuple(); device, backend)

Sets up `fun` for just-in-time compilation with XLA.

Arguments:
  - `fun`: Function to be jitted. Should be a pure function, as side-effects may
    only be executed once. Its arguments and return value should be arrays,
    scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
    Positional arguments indicated by `static_argnums` can be anything at
    all, provided they are hashable and have an equality operation defined.
    Static arguments are included as part of a compilation cache key, which is
    why hash and equality operators must be defined.
  - `static_argnums`: An int or tuple of ints specifying which positional
    arguments to treat as static (compile-time constant). Operations that only
    depend on static arguments will be constant-folded. Calling the jitted
    function with different values for these constants will trigger
    recompilation. If the jitted function is called with fewer positional
    arguments than indicated by `static_argnums` then an error is raised.
    Defaults to ().

  - `device`: This is an experimental feature and the API is likely to change.
    Optional, the Device the jitted function will run on. (Available devices
    can be retrieved via `jax.devices()`.) The default is inherited from
    XLA's DeviceAssignment logic and is usually to use ``jax.devices()[0]``.

  - `backend`: This is an experimental feature and the API is likely to change.
    Optional, a string representing the xla backend. 'cpu','gpu', or 'tpu'.

Returns:
  A wrapped version of ``fun``, set up for just-in-time compilation.
"""
jit(f; static_argnums=tuple(), device=nothing, backend=nothing) = begin
    jit(f, static_argnums...; device=device, backend=backend)
end

function jit(f, s, static_argnums...; device=nothing, backend=nothing)
    # Change from Julia's 1 based to Python's 0 based indexing
    s = tuple(s..., static_argnums...)
    _static_argnums = s .- 1
    return jax.jit(f, static_argnums=_static_argnums, device=device, backend=backend)
end


##

"""
    grad(f, argnums; has_aux=false, holomorphic=false)

Creates a function which evaluates the gradient of ``fun``.

Args:
  - fun: Function to be differentiated. Its arguments at positions specified by
    ``argnums`` should be arrays, scalars, or standard Python containers. It
    should return a scalar (which includes 0-dimensional arrays but not
    1-dimensional arrays of size 1.)
  - argnums: Optional, integer or sequence of integers. Specifies which
    positional argument(s) to differentiate with respect to (default 0).
  - has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
    first element is considered the output of the mathematical function to be
    differentiated and the second element is auxiliary data. Default False.
  - holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
    holomorphic. Default False.

Returns:
  A function with the same arguments as ``fun``, that evaluates the gradient
  of ``fun``. If ``argnums`` is an integer then the gradient has the same
  shape and type as the positional argument indicated by that integer. If
  argnums is a tuple of integers, the gradient is a tuple of values with the
  same shapes and types as the corresponding arguments. If ``has_aux`` is True
  then a pair of (gradient, auxiliary_data) is returned.

For example:

>>> grad_tanh = jax.grad(jax.numpy.tanh)
>>> print(grad_tanh(0.2))
0.961043
"""
grad(f; argnums=tuple(), has_aux=false, holomorphic=false) = begin
    grad(f, static_argnums...; device=device, backend=backend)
end

function grad(f, s, argnums...; has_aux=false, holomorphic=false)
    # Change from Julia's 1 based to Python's 0 based indexing
    s = tuple(s..., argnums...)
    _argnums = s .- 1
    return jax.grad(f, argnums=_argnums, has_aux=has_aux, holomorphic=holomorphic)
end
