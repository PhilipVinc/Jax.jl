using Statistics

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
