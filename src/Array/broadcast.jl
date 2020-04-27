using Base: Broadcast

struct JaxArrayStyle{N} <: JaxAbstractArrayStyle{N} end

JaxArrayStyle(::Val{N}) where {N} = JaxArrayStyle{N}()
JaxArrayStyle{M}(::Val{N}) where {N,M} = JaxArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:JaxArray{T,N}}) where {T,N} =
    JaxArrayStyle{N}()

Broadcast.materialize(bc::Broadcast.Broadcasted{<:JaxArrayStyle}) =
    convert(JaxArray, _materialize(bc))
