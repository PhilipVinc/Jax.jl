
mutable struct JaxArray{T,N} <: AbstractJaxArray{T,N}
  o::PyObject
  dims::NTuple{N,Int}
end

const JaxVector{T} = JaxArray{T,1}
const JaxMatrix{T} = JaxArray{T,2}

JaxArray(arr::AbstractArray) = JaxArray(collect(arr))
function JaxArray(arr::Array{T,N}) where {T,N}
  # When converting an array to Jax Array we transpose it.
  arr = PyReverseDims(arr)
  return numpy.array(arr)
end

function JaxArray(parr::PyObject)
  if !pyisinstance(parr, jax.interpreters.xla.DeviceArray)
    error("It is not jax.interpreters.xla.DeviceArray but $parr ")
  end
  T = np_to_jl_type(parr.dtype)
  return JaxArray{T,length(parr.shape)}(parr, reverse(parr.shape))
end

JaxArray(d...) = JaxArray(Float32, d...)
JaxArray(T::Type{<:Number}, d...) = JaxArray(T, undef, d...)
JaxArray(T::Type{<:Number}, u::UndefInitializer, d::Integer...) =
  JaxArray(T, u, Dims(d))
function JaxArray(T::Type{<:Number}, u::UndefInitializer, d::Dims)
  return numpy.empty(reverse(d), dtype = jl_to_np_type(T))#JaxArray()
end

Base.size(a::JaxArray) = a.dims
Base.length(a::JaxArray) = prod(a.dims)
Base.eltype(a::JaxArray{T}) where {T} = T
Base.axes(a::JaxArray) = map(Base.OneTo, size(a))

PyCall.PyObject(a::JaxArray) = a.o
Base.convert(::Type{<:JaxArray}, o::PyObject) = JaxArray(o)

_transpose(x::Union{AbstractVector,AbstractMatrix}) = transpose(x)
_transpose(x::AbstractArray{T,N}) where {T,N} = permutedims(x, reverse(1:N))

_values(a::JaxArray) = PyArray_ReadOnly(a.o._value) |>_transpose
_values(a::JaxArray{T,0}) where T = PyArray_ReadOnly(a.o._value)
_values(a::JaxArray{T,1}) where T = PyArray_ReadOnly(a.o._value)
Base.collect(a::JaxArray) = copy(_values(a))
Base.convert(::Type{<:Array}, a::JaxArray) = copy(_values(a))

# show
_szstr(d::Tuple{}) = "$(0)-dimensional"
_szstr(d::Tuple{Int}) = "$(first(d))-element"
_szstr(dims::Dims) = prod(["$i×" for i in dims])[1:end-1]
function Base.show(io::IO, m::MIME"text/plain", a::JaxArray{T,N}) where {T,N}
  print(io, "$(_szstr(a.dims)) JaxArray{$T,$N}")
  isempty(a) && return
  print(io, ":")

  if !haskey(io, :compact) && length(axes(a, 2)) > 1
      io = IOContext(io, :compact => true)
  end
  if get(io, :limit, false) && eltype(a) === Method
      # override usual show method for Vector{Method}: don't abbreviate long lists
      io = IOContext(io, :limit => false)
  end

  if get(io, :limit, false) && displaysize(io)[1]-4 <= 0
      return print(io, " …")
  else
      println(io)
  end

  io = IOContext(io, :typeinfo => eltype(a))

  Base.print_array(io, _values(a))
end

function Base.show(io::IO, a::JaxArray{T,N}) where {T,N}
  print(io, "$(_szstr(a.dims)) JaxArray{$T,$N}:")
  vals = _values(a)
  if N == 0
    print(io, "[$(first(vals))]\n")
  else
    print(io, "[$(first(vals)) … $(last(vals))]\n")
  end
end

# very slow
# Base.getindex(a::JaxArray, args...) = (getindex(_values(a), args...))

# iterators
Base.eachrow(a::JaxArray) = (get(a.o, i-1) for i in axes(a, 1))

block_until_ready(a::JaxArray) = a.o.block_until_ready()
