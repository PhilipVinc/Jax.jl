
mutable struct ParallelTracerArray{T,N} <: AbstractJaxArray{T,N}
  o::PyObject
  dims::NTuple{N,Int}
end

const ParallelTracerVector{T} = TracedArray{T,1}
const ParallelTracerMatrix{T} = TracedArray{T,2}

function ParallelTracerArray(parr::PyObject)
  @assert pyisinstance(parr, jax.interpreters.parallel.PapplyTracer)
  aval = parr.aval
  T = np_to_jl_type(aval.dtype)
  return ParallelTracerArray{T,length(aval.shape)}(parr, reverse(aval.shape))
end

Base.size(a::ParallelTracerArray) = a.dims
Base.length(a::ParallelTracerArray) = prod(a.dims)
Base.eltype(a::ParallelTracerArray{T}) where {T} = T
Base.axes(a::ParallelTracerArray) = map(Base.OneTo, size(a))

PyCall.PyObject(a::ParallelTracerArray) = a.o
Base.convert(::Type{<:ParallelTracerArray}, o::PyObject) = ParallelTracerArray(o)

function Base.transpose(a::ParallelTracerArray{T,N}) where {T,N}
  po = a.o.transpose()
  return ParallelTracerArray{T,N}(po, po.aval.shape)
end

function Base.conj(a::ParallelTracerArray{T,N}) where {T,N}
  ParallelTracerArray{T,N}(numpy.conj(a.o), size(a))
end

function Base.adjoint(a::ParallelTracerArray{T,N}) where {T,N}
  po = a.o.transpose().conj()
  ParallelTracerArray{T,N}(po, po.aval.shape)
end

Base.adjoint(a::ParallelTracerArray{T}) where {T<:Real} = transpose(a)

function Base.show(io::IO, ::MIME"text/plain", a::ParallelTracerArray{T,N}) where {T,N}
  println(io, "$(_szstr(a.dims)) ParallelTracerArray{JaxArray{$T,$N}}:")
  print(io, " ", a.o)
end

function Base.show(io::IO, a::ParallelTracerArray{T,N}) where {T,N}
  print(io, "$(_szstr(a.dims)) ParallelTracerArray{JaxArray{$T,$N}}")
end
