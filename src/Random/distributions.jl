
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
  _random.uniform(key, reverse(dims), jl_to_np_type(T); minval = minval, maxval = maxval)
end

function uniform(
  T::Type{<:Integer},
  key::JaxRNGKey,
  dims::Dims;
  minval::Integer = 0,
  maxval::Integer = 1,
)
  _random.randint(
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
  _random.normal(key, reverse(dims), dtype = jl_to_np_type(T))
end
