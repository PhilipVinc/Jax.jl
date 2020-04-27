Random.rand(r::JaxRNGKey, dims::Integer...) = rand(r, Float64, Dims(dims))
Random.rand(r::JaxRNGKey, T::Type, dims::Dims)  = uniform(T, r, dims, minval=zero(T), maxval=one(T))
Random.rand(r::JaxRNGKey, T::Type{<:Integer}, dims::Dims)  = uniform(T, r, dims, minval=typemin(T), maxval=typemax(T))

Random.randn(r::JaxRNGKey, dims::Integer...) = rand(r, Float64, Dims(dims))
Random.randn(r::JaxRNGKey, T::Type, dims::Dims)  = normal(T, r, dims)



#=
Random.rand(r::JaxRNGKey, X, d::Integer, dims::Integer...) = rand(r, X, Dims((d, dims...)))

Random.rand(r::JaxRNGKey, ::Type{X}, dims::Dims) where {X} = rand!(r, Array{X}(undef, dims), X)

Random.rand(r::JaxRNGKey, ::Type{X}, d::Integer, dims::Integer...) where {X} = rand(r, X, Dims((d, dims...)))
=#
