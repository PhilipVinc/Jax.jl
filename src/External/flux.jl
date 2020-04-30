using .Flux: Flux, fmap, functor

# like gpu() from flux
tojax(m) = fmap(x -> adapt(JaxArray, x), m)

# Jax will recursively call this on any jl object (like jl_wrap<identity>)
# until he gets only fundamental types out. THankfully, that's mostly
# what functor does.
_flatten(x) = begin
    d, f = functor(x)
    tup = tuple(d...)
    #@info "Called Chain_flatten: $x" tup

    return tup, f
end

_unflatten(f, x) = begin
    #@info("called chain_unf $f - $x")
    return f(x)
end

jl_flatten(x) = _flatten(x)
jl_unflatten(f, x) = _unflatten(f, x)

# This is an horrinble hack.
# it will register as a pytree node any wrapper of julia
# objects.
# It woudl be nice to internally dispatch on traits when calling
TreeUtil.register_pytree_node(
    pytypeof(PyObject(identity)),
    jl_flatten,
    jl_unflatten,
)

Jax.@jaxfunc Flux.leakyrelu(x) = Jax.jax.nn.leaky_relu(x)
Jax.@jaxfunc Flux.relu(x) = Jax.jax.nn.relu(x)
Jax.@jaxfunc Flux.gelu(x) = Jax.jax.nn.gelu(x)
Jax.@jaxfunc Flux.elu(x) = Jax.jax.nn.elu(x)
Jax.@jaxfunc Flux.sigmoid(x) = Jax.jax.nn.sigmoid(x)
Jax.@jaxfunc Flux.logsigmoid(x) = Jax.jax.nn.log_sigmoid(x)
Jax.@jaxfunc Flux.selu(x) = Jax.jax.nn.selu(x)
Jax.@jaxfunc Flux.swish(x) = Jax.jax.nn.swish(x)
