
struct TracedArray{T,N} <: AbstractJaxArray{T,N}
    o::PyObject
    dims::NTuple{N,Int}
end

const TracedVector{T} = TracedArray{T,1}
const TracedMatrix{T} = TracedArray{T,2}

function TracedArray(parr::PyObject)
  @assert pyisinstance(parr, jax.interpreters.partial_eval.JaxprTracer)
  aval = parr.aval
  T = np_to_jl_type(aval.dtype)
  return TracedArray{T,length(aval.shape)}(parr, aval.shape)
end

#TracedArray(d...) = TracedArray(Float32, d...)
#TracedArray(T::Type{<:Number}, d...) = TracedArray(T, undef, d...)
#TracedArray(T::Type{<:Number}, u::UndefInitializer, d::Integer...) =
#  TracedArray(T, u, Dims(d))
#function TracedArray(T::Type{<:Number}, u::UndefInitializer, d::Dims)
#  return np.empty(d, dtype = jl_to_np_type(T))#JaxArray()
#end

Base.size(a::TracedArray) = a.dims
Base.length(a::TracedArray) = prod(a.dims)
Base.eltype(a::TracedArray{T}) where {T} = T
Base.axes(a::TracedArray) = map(Base.OneTo, size(a))

PyCall.PyObject(a::TracedArray) = a.o
Base.convert(::Type{<:TracedArray}, o::PyObject) = TracedArray(o)

function Base.transpose(a::TracedArray{T,N}) where {T,N}
  po = a.o.transpose()
  return TracedArray{T,N}(po, po.aval.shape)
end

function Base.conj(a::TracedArray{T,N}) where {T,N}
  TracedArray{T,N}(np.conj(a.o), size(a))
end

function Base.adjoint(a::TracedArray{T,N}) where {T,N}
  po = a.o.transpose().conj()
  TracedArray{T,N}(po, po.aval.shape)
end

Base.adjoint(a::TracedArray{T}) where {T<:Real} = transpose(a)

function Base.show(io::IO, ::MIME"text/plain", a::TracedArray{T,N}) where {T,N}
  println(io, "$(_szstr(a.dims)) Traced{JaxArray{$T,$N}}:")
  print(io, " ",a.o)
end

function Base.show(io::IO, a::TracedArray{T,N}) where {T,N}
  print(io, "$(_szstr(a.dims)) Traced{JaxArray{$T,$N}}")
end
