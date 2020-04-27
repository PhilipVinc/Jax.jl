function Base.zero(a::T) where T<:AbstractJaxArray
    return JaxArray(eltype(T), size(a))
end
