using Functors: fmap, functor

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
