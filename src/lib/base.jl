function Base.zero(a::T) where T<:AbstractJaxArray
    return JaxArray(eltype(T), size(a))
end

Base.conj(x::AbstractJaxArray) = numpy.conj(x)
Base.conj(x::AbstractJaxArray{T}) where {T<:Real} = x
