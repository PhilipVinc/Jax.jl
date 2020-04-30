using Base: Broadcast

struct DefaultJaxArrayStyle{N} <: JaxAbstractArrayStyle{N} end
DefaultJaxArrayStyle(::Val{N}) where {N} = DefaultJaxArrayStyle{N}()
DefaultJaxArrayStyle{M}(::Val{N}) where {N,M} = DefaultJaxArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:AbstractJaxArray{T,N}}) where {T,N} =
    DefaultJaxArrayStyle{N}()

_BroadcastStyle(::Type{<:AbstractArray{T,N}}) where {T,N} = DefaultJaxArrayStyle{N}()
Base.convert(::Type{<:DefaultJaxArrayStyle}, a::Broadcast.DefaultArrayStyle{M}) where M = DefaultJaxArrayStyle{M}()

# When broadcasting with Jax replace Julia functions with jax functions
Base.broadcasted(::JAS, f, args...) where {JAS<:JaxAbstractArrayStyle} =
    Broadcast.Broadcasted{JAS}(jaxfunc(f), args, nothing)

_pyconvert(o::PyObject) = convert(PyAny, o)
_pyconvert(o::AbstractJaxArray) = o

# Outermost layer of materialization. Convert the result of recursive inner
# _materialize calls
Broadcast.materialize(bc::Broadcast.Broadcasted{<:JaxAbstractArrayStyle}) =
    _pyconvert(_materialize(Broadcast.instantiate(bc)))

# Standard things, go back to Base
_materialize(bc::Broadcast.Broadcasted) = Base.materialize(bc)

# Numbers and other objects, just return them
_materialize(bc) = bc

# Jax stuff, go to python
_materialize(bc::Broadcast.Broadcasted{<:JaxAbstractArrayStyle}) = __pymaterialize(Broadcast.instantiate(bc))

__pymaterialize(bc::Broadcast.Broadcasted)  = bc.f(map(_materialize, bc.args)...)

# Jax correctly preserves 0-dim objects
Base.broadcast_preserving_zero_d(f, As::AbstractJaxArray...) =
    Broadcast.materialize(Base.broadcasted(f, As...))

## inplace
function Base.copyto!(
    dest::AbstractJaxArray{T,N},
    bc::Broadcast.Broadcasted{Nothing},
) where {T,N}
    obj = __pymaterialize(bc)

    indices = N!=0 ? jax.ops.index[:] : nothing

    o = pycall(
        jax.ops.index_update,
        PyObject,
        dest,
        indices,
        obj,
    )

    # Swap the inner buffer because jax does not really do in-place mutation
    dest.o = o
    return dest
end
