using Statistics

_invert_dim(N, i) = N-i

_convert_dims(N::Integer, d::Integer) = N-d
_convert_dims(N::Integer, d::Dims)    = N.-i #map(i->_invert_dim(N, i), d)
_convert_dims(N::Integer, d::Integer...) = _convert_dims(N, Dims(d))
_convert_dims(N::Integer, d::Colon) = nothing

_convert_dims(::Dims{N}, d) where N = _convert_dims(N, d)
_convert_dims(::AbstractArray{T,N}, d) where {T,N} = _convert_dims(N, d)

_autokeep(dims::Colon) = false
_autokeep(dims) = true

for (op, fun) in ((:sum, :sum), (:mean, :mean))
    isdefined(Statistics, op) || continue
    _op = Symbol("_", op)
    __op = Symbol("__", op)

    @eval begin
        Statistics.$op(x::AbstractJaxArray; dims=:, keepdims=_autokeep(dims)) = $_op(nothing, x, keepdims, dims)
        Statistics.$op(T::Type, x::AbstractJaxArray; dims=:, keepdims=_autokeep(dims)) = $_op(T, x, keepdims, dims)

        $_op(T, x, keepdims, dims...) = $__op(T, x, keepdims, _convert_dims(size(x), dims...))
        $__op(T, x, keepdims, dims) = numpy.$fun(x, axis=dims, dtype=T, keepdims=keepdims)
    end
end
