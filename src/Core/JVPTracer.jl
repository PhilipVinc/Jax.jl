
mutable struct JVPTracerArray{T,N} <: AbstractJaxArray{T,N}
  o::PyObject
  dims::NTuple{N,Int}
end

const JVPTracerVector{T} = TracedArray{T,1}
const JVPTracerMatrix{T} = TracedArray{T,2}

function JVPTracerArray(parr::PyObject)
  @assert pyisinstance(parr, jax.interpreters.ad.JVPTracer)
  aval = parr.aval
  T = np_to_jl_type(aval.dtype)
  return JVPTracerArray{T,length(aval.shape)}(parr, reverse(aval.shape))
end

Base.size(a::JVPTracerArray) = a.dims
Base.length(a::JVPTracerArray) = prod(a.dims)
Base.eltype(a::JVPTracerArray{T}) where {T} = T
Base.axes(a::JVPTracerArray) = map(Base.OneTo, size(a))

PyCall.PyObject(a::JVPTracerArray) = a.o
Base.convert(::Type{<:JVPTracerArray}, o::PyObject) = JVPTracerArray(o)

function Base.show(io::IO, ::MIME"text/plain", a::JVPTracerArray{T,N}) where {T,N}
  println(io, "$(_szstr(a.dims)) JVPTracerArray{JaxArray{$T,$N}}:")
  print(io, " ", a.o)
end

function Base.show(io::IO, a::JVPTracerArray{T,N}) where {T,N}
  print(io, "$(_szstr(a.dims)) JVPTracerArray{JaxArray{$T,$N}}")
end
