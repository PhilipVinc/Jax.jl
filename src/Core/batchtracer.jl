
mutable struct BatchTracerArray{T,N} <: AbstractJaxArray{T,N}
  o::PyObject
  dims::NTuple{N,Int}
end

const BatchTracerVector{T} = TracedArray{T,1}
const BatchTracerMatrix{T} = TracedArray{T,2}

function BatchTracerArray(parr::PyObject)
  @assert pyisinstance(parr, jax.interpreters.batching.BatchTracer)
  aval = parr.aval
  T = np_to_jl_type(aval.dtype)
  return BatchTracerArray{T,length(aval.shape)}(parr, reverse(aval.shape))
end

Base.size(a::BatchTracerArray) = a.dims
Base.length(a::BatchTracerArray) = prod(a.dims)
Base.eltype(a::BatchTracerArray{T}) where {T} = T
Base.axes(a::BatchTracerArray) = map(Base.OneTo, size(a))

PyCall.PyObject(a::BatchTracerArray) = a.o
Base.convert(::Type{<:BatchTracerArray}, o::PyObject) = BatchTracerArray(o)

function Base.transpose(a::BatchTracerArray{T,N}) where {T,N}
  po = a.o.transpose()
  return BatchTracerArray{T,N}(po, po.aval.shape)
end

function Base.conj(a::BatchTracerArray{T,N}) where {T,N}
  BatchTracerArray{T,N}(numpy.conj(a.o), size(a))
end

function Base.adjoint(a::BatchTracerArray{T,N}) where {T,N}
  po = a.o.transpose().conj()
  BatchTracerArray{T,N}(po, po.aval.shape)
end

Base.adjoint(a::BatchTracerArray{T}) where {T<:Real} = transpose(a)

function Base.show(io::IO, ::MIME"text/plain", a::BatchTracerArray{T,N}) where {T,N}
  println(io, "$(_szstr(a.dims)) BatchTracerArray{JaxArray{$T,$N}}:")
  print(io, " ", a.o)
end

function Base.show(io::IO, a::BatchTracerArray{T,N}) where {T,N}
  print(io, "$(_szstr(a.dims)) BatchTracerArray{JaxArray{$T,$N}}")
end
