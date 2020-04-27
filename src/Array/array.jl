
struct JaxArray{T,N} <: AbstractJaxArray{T,N}
  o::PyObject
  dims::NTuple{N,Int}
end

const JaxVector{T} = JaxArray{T,1}
const JaxMatrix{T} = JaxArray{T,2}

function JaxArray(arr::Array{T,N}) where {T,N}
  return np.array(arr)
end

function JaxArray(parr::PyObject)
  @assert pyisinstance(parr, jax.interpreters.xla.DeviceArray)
  T = np_to_jl_type(parr.dtype)
  return JaxArray{T,length(parr.shape)}(parr, parr.shape)
end

JaxArray(d...) = JaxArray(Float32, d...)
JaxArray(T::Type{<:Number}, d...) = JaxArray(T, undef, d...)
JaxArray(T::Type{<:Number}, u::UndefInitializer, d::Integer...) =
  JaxArray(T, u, Dims(d))
function JaxArray(T::Type{<:Number}, u::UndefInitializer, d::Dims)
  return np.empty(d, dtype = jl_to_np_type(T))#JaxArray()
end

Base.size(a::JaxArray) = a.dims
Base.length(a::JaxArray) = prod(a.dims)
Base.eltype(a::JaxArray{T}) where {T} = T
Base.axes(a::JaxArray) = map(Base.OneTo, size(a))

PyCall.PyObject(a::JaxArray) = a.o
Base.convert(::Type{<:JaxArray}, o::PyObject) = JaxArray(o)

_values(a::JaxArray) = PyArray_ReadOnly(a.o._value)
Base.collect(a::JaxArray) = copy(_values(a))

# linalg special
function Base.transpose(a::JaxArray{T,N}) where {T,N}
  po = a.o.transpose()
  return JaxArray{T,N}(po, po.shape)
end

function Base.conj(a::JaxArray{T,N}) where {T,N}
  JaxArray{T,N}(np.conj(a.o), size(a))
end

function Base.adjoint(a::JaxArray{T,N}) where {T,N}
  po = a.o.transpose().conj()
  JaxArray{T,N}(po, po.shape)
end

Base.adjoint(a::JaxArray{T}) where {T<:Real} = transpose(a)

# show
_szstr(d::Tuple{}) = "$(0)-dimensional"
_szstr(d::Tuple{Int}) = "$(first(d))-element"
_szstr(dims::Dims) = prod(["$i×" for i in dims])[1:end-1]
function Base.show(io::IO, m::MIME"text/plain", a::JaxArray{T,N}) where {T,N}
  println(io, "$(_szstr(a.dims)) JaxArray{$T,$N}:")
  Base.print_array(io, _values(a))
end

function Base.show(io::IO, a::JaxArray{T,N}) where {T,N}
  print(io, "$(_szstr(a.dims)) JaxArray{$T,$N}:")
  vals = _values(a)
  print("[$(first(vals)) … $(last(vals))]")
end

# very slow
# Base.getindex(a::JaxArray, args...) = (getindex(_values(a), args...))

# iterators
Base.eachrow(a::JaxArray) = (get(a.o, i-1) for i in axes(a, 1))
