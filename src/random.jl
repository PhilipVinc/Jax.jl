using Random

struct JaxRNGKey <: AbstractRNG
  o::PyObject
end

PRNGKey(seed::Integer) =
  JaxRNGKey(PyCall.pycall(random.PRNGKey, PyObject, seed))
PRNGKey(o::PyObject) = JaxRNGKey(o)

PyCall.PyObject(rng::JaxRNGKey) = rng.o
Base.convert(::Type{JaxRNGKey}, o::PyObject) = JaxRNGKey(o)

_values(a::JaxRNGKey) = PyArray_ReadOnly(a.o._value)

function Base.show(io::IO, m::MIME"text/plain", k::JaxRNGKey)
  print(io, "Jax PRNG Key ")
  print(io, collect(_values(k)))
end

### Methods

function fold_in(key::JaxRNGKey, data::Integer)
  o = PyCall.pycall(random.fold_in, PyObject, key, data)
  return JaxRNGKey(o)
end

function split(key::JaxRNGKey, num = 2)
  o_keys = PyCall.pycall(random.split, PyObject, key, num)
  keys = [JaxRNGKey(get(o_keys, PyObject, i)) for i = 1:length(o_keys)]
  return keys
end

#jax.random.threefry_2x32
uniform(key::JaxRNGKey, dims...; kwargs...) =
  uniform(Float64, key, dims...; kwargs...)
uniform(T::Type{<:Number}, key::JaxRNGKey, dims...; kwargs...) =
  uniform(T, key, Dims(dims); kwargs...)
function uniform(
  T::Type{<:Number},
  key::JaxRNGKey,
  dims::Dims;
  minval = 0.0,
  maxval = 1.0,
)
  random.uniform(key, dims, jl_to_np_type(T); minval = minval, maxval = maxval)
end
function uniform(
  T::Type{<:Integer},
  key::JaxRNGKey,
  dims::Dims;
  minval::Integer = 0,
  maxval::Integer = 1,
)
  random.randint(
    key,
    dims,
    dtype = jl_to_np_type(T),
    minval = minval,
    maxval = maxval,
  )
end


normal(key::JaxRNGKey, dims...; kwargs...) = uniform(Float64, key, dims...)
normal(T::Type{<:Number}, key::JaxRNGKey, dims...) = normal(T, key, Dims(dims))
function normal(T::Type{<:Number}, key::JaxRNGKey, dims::Dims)
  random.normal(key, dims, dtype = jl_to_np_type(T))
end
