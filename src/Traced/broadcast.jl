using Base: Broadcast

struct TracedArrayStyle{N} <: JaxAbstractArrayStyle{N} end

TracedArrayStyle(::Val{N}) where {N} = TracedArrayStyle{N}()
TracedArrayStyle{M}(::Val{N}) where {N,M} = TracedArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:TracedArray{T,N}}) where {T,N} =
    TracedArrayStyle{N}()

Broadcast.materialize(bc::Broadcast.Broadcasted{<:TracedArrayStyle}) =
    convert(TracedArray, _materialize(bc))
