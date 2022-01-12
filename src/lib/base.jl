function Base.zero(a::T) where T<:AbstractJaxArray
    return JaxArray(eltype(T), size(a))
end

Base.conj(x::AbstractJaxArray) = numpy.conj(x)
Base.conj(x::AbstractJaxArray{T}) where {T<:Real} = x

Base.real(x::AbstractJaxArray) = numpy.real(x)
Base.real(x::AbstractJaxArray{T}) where {T<:Real} = x

Base.imag(x::AbstractJaxArray) = numpy.imag(x)
Base.imag(x::AbstractJaxArray{T}) where {T<:Real} = x
