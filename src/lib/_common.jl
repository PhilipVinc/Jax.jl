_convert_dims(N::Integer, d::Integer) = N-d
_convert_dims(N::Integer, d::Dims)    = N.-d #map(i->_invert_dim(N, i), d)
_convert_dims(N::Integer, d::Integer...) = _convert_dims(N, Dims(d))
_convert_dims(N::Integer, d::Colon) = nothing

_convert_dims(::Dims{N}, d) where N = _convert_dims(N, d)
_convert_dims(::AbstractArray{T,N}, d) where {T,N} = _convert_dims(N, d)

_autokeep(dims::Colon) = false
_autokeep(dims) = true
