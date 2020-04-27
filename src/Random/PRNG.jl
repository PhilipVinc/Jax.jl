struct JaxRNGKey <: AbstractRNG
  o::PyObject
end

PRNGKey(seed::Integer) =
  JaxRNGKey(PyCall.pycall(_random.PRNGKey, PyObject, seed))
PRNGKey(o::PyObject) = JaxRNGKey(o)

PyCall.PyObject(rng::JaxRNGKey) = rng.o
Base.convert(::Type{JaxRNGKey}, o::PyObject) = JaxRNGKey(o)

_values(a::JaxRNGKey) = PyArray(a.o._value, true)

function Base.show(io::IO, m::MIME"text/plain", k::JaxRNGKey)
  print(io, "Jax PRNG Key ")
  print(io, collect(_values(k)))
end

### Methods operating on keys
function fold_in(key::JaxRNGKey, data::Integer)
  o = PyCall.pycall(_random.fold_in, PyObject, key, data)
  return JaxRNGKey(o)
end

function split(key::JaxRNGKey, num = 2)
  o_keys = PyCall.pycall(_random.split, PyObject, key, num)
  keys = [JaxRNGKey(get(o_keys, PyObject, i)) for i = 1:length(o_keys)]
  return keys
end

#jax._random.threefry_2x32
