using Base: Broadcast

struct ParallelTracerArrayStyle{N} <: JaxAbstractArrayStyle{N} end

ParallelTracerArrayStyle(::Val{N}) where {N} = ParallelTracerArrayStyle{N}()
ParallelTracerArrayStyle{M}(::Val{N}) where {N,M} = ParallelTracerArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:ParallelTracerArray{T,N}}) where {T,N} =
    ParallelTracerArrayStyle{N}()

Broadcast.materialize(bc::Broadcast.Broadcasted{<:ParallelTracerArrayStyle}) =
    convert(ParallelTracerArray, _materialize(bc))
