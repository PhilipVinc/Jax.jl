using Base: Broadcast

struct JVPTracerArrayStyle{N} <: JaxAbstractArrayStyle{N} end

JVPTracerArrayStyle(::Val{N}) where {N} = JVPTracerArrayStyle{N}()
JVPTracerArrayStyle{M}(::Val{N}) where {N,M} = JVPTracerArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:JVPTracerArray{T,N}}) where {T,N} =
    JVPTracerArrayStyle{N}()

Broadcast.materialize(bc::Broadcast.Broadcasted{<:JVPTracerArrayStyle}) =
    convert(JVPTracerArray, _materialize(bc))
