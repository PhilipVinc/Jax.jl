using Base: Broadcast

struct BatchTracerArrayStyle{N} <: JaxAbstractArrayStyle{N} end

BatchTracerArrayStyle(::Val{N}) where {N} = BatchTracerArrayStyle{N}()
BatchTracerArrayStyle{M}(::Val{N}) where {N,M} = BatchTracerArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:BatchTracerArray{T,N}}) where {T,N} =
    BatchTracerArrayStyle{N}()

Broadcast.materialize(bc::Broadcast.Broadcasted{<:BatchTracerArrayStyle}) =
    convert(BatchTracerArray, _materialize(bc))

Base.BroadcastStyle(a::BatchTracerArrayStyle{Any}, ::JaxArrayStyle) = a
Base.BroadcastStyle(a::BatchTracerArrayStyle{N}, ::JaxArrayStyle{N}) where N = a
Base.BroadcastStyle(a::BatchTracerArrayStyle{M}, ::JaxArrayStyle{N}) where {M,N} = typeof(a)(Val(max(M, N)))
