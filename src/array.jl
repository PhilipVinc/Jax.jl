
struct JaxArray{T,N} <: AbstractArray{T,N}
  o::PyObject
  dims::NTuple{N,Int}
end

function JaxArray(arr::Array{T,N}) where {T,N}
  return JaxArray{Float32,N}(
    np.array(convert(Array{Float32,N}, arr)),
    size(arr),
  )
end

function JaxArray(parr::PyObject)
  @assert pyisinstance(parr, jax.interpreters.xla.DeviceArray)
  T = np_to_jl_type(parr.dtype)
  return JaxArray{T,length(parr.shape)}(parr, parr.shape)
end

JaxArray(T::Type{<:Number}, d...) = JaxArray(T, undef, d...)
JaxArray(T::Type{<:Number}, u::UndefInitializer, d::Integer...) =
  JaxArray(T, u, Dims(d))
function JaxArray(T::Type{<:Number}, u::UndefInitializer, d::Dims)
  return JaxArray(np.empty(d, dtype = jl_to_np_type(T)))
end

Base.size(a::JaxArray) = a.dims
Base.length(a::JaxArray) = prod(a.dims)
Base.eltype(a::JaxArray{T}) where {T} = T
Base.axes(a::JaxArray) = map(Base.OneTo, size(a))

PyCall.PyObject(a::JaxArray) = a.o
Base.convert(::Type{JaxArray}, o::PyObject) =
  JaxArray{Float32,length(o.shape)}(o, o.shape)


# linalg special
function Base.transpose(a::JaxArray{T,N}) where {T,N}
  po = a.o.transpose()
  return JaxArray{T,N}(po, po.shape)
end

function Base.conj(a::JaxArray{T,N}) where {T,N}
  JaxArray{T,N}(np.conj(a.o), size(a))
end

# show

_szstr(d::Tuple{}) = "$(0)-dimensional"
_szstr(d::Tuple{Int}) = "$(first(d))-element"
_szstr(dims::Dims) = prod(["$i×" for i in dims])[1:end-1]
function Base.show(io::IO, m::MIME"text/plain", a::JaxArray{T,N}) where {T,N}
  println(io, "$(_szstr(a.dims)) JaxArray{$T,$N}:")
  print(io, a.o)
end
