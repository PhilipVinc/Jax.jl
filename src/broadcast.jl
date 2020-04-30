using Base: Broadcast

struct DefaultJaxArrayStyle{N} <: JaxAbstractArrayStyle{N} end
DefaultJaxArrayStyle(::Val{N}) where {N} = DefaultJaxArrayStyle{N}()
DefaultJaxArrayStyle{M}(::Val{N}) where {N,M} = DefaultJaxArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:AbstractJaxArray{T,N}}) where {T,N} =
    DefaultJaxArrayStyle{N}()

_BroadcastStyle(::Type{<:AbstractArray{T,N}}) where {T,N} =
    DefaultJaxArrayStyle{N}()
Base.convert(
    ::Type{<:DefaultJaxArrayStyle},
    a::Broadcast.DefaultArrayStyle{M},
) where {M} = DefaultJaxArrayStyle{M}()

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
_materialize(bc::Broadcast.Broadcasted{<:JaxAbstractArrayStyle}) =
    __pymaterialize(Broadcast.instantiate(bc))

__pymaterialize(bc::Broadcast.Broadcasted) = bc.f(map(_materialize, bc.args)...)

# Jax correctly preserves 0-dim objects
Base.broadcast_preserving_zero_d(f, As::AbstractJaxArray...) =
    Broadcast.materialize(Base.broadcasted(f, As...))

## inplace
Base.copyto!(dest::AbstractJaxArray, bc::Broadcast.Broadcasted{Nothing}) =
    _copyto!(dest, bc)

Base.copyto!(
    dest::AbstractJaxArray,
    bc::Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
) = _copyto!(dest, bc)

function _copyto!(dest::AbstractJaxArray{T,N}, bc::Broadcast.Broadcasted) where {T,N}
    obj = __pymaterialize(bc)

    indices = N != 0 ? jax.ops.index[:] : nothing

    o = pycall(jax.ops.index_update, PyObject, dest, indices, obj)

    # Swap the inner buffer because jax does not really do in-place mutation
    dest.o = o
    return dest
end

#=
## wrappers
using LinearAlgebra
for (W, ctor) in Adapt.wrappers
  @eval begin
    Broadcast.BroadcastStyle(T::Type{<:$W}) where {AT<:AbstractJaxArray} = _BroadcastStyle(T)
    #backend(::Type{<:$W}) where {AT<:AbstractGPUArray} = backend(AT)
  end
end

function _materialize(wa::LinearAlgebra.Transpose{T,A}) where {T,N,A<:AbstractJaxArray{T,N}}
    if N==1
        numpy.reshape(parent(wa), (-1,1))
    elseif N==2
        numpy.transpose(parent(wa))
    else
        error("Transpose not supported")
    end
end

_materialize(wa::LinearAlgebra.Adjoint) = conj(_materialize(Transpose(parent(wa))))

#=_materialize(bc::Broadcast.Broadcasted{<:JaxAbstractArrayStyle, T, typeof(_pmul), Tuple{V1,V2}}) where {T, V1<:AbstractJaxVector, V2<:AdjOrTrans} = begin
    @info "Jad._materialize{JaxAbstractArrayStyle} t" typeof(bc) bc.f bc.axes typeof(bc.args)
    arg1 = _materialize(bc.args[1])
    arg2 = _materialize(bc.args[2])
    #args = map(_materialize, bc.args...)
    numpy.outer(arg2, arg1)
end

_materialize(bc::Broadcast.Broadcasted{<:JaxAbstractArrayStyle, T, typeof(_pmul), Tuple{V1,V2}}) where {T, V1<:AdjOrTrans, V2<:AbstractJaxVector} = begin
    @info "Jad._materialize{JaxAbstractArrayStyle} t" typeof(bc) bc.f bc.axes typeof(bc.args)
    arg1 = _materialize(bc.args[1])
    arg2 = _materialize(bc.args[2])
    #args = map(_materialize, bc.args...)
    numpy.outer(arg1, arg2)
end
=#
=#
