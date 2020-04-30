Base.transpose(a::AbstractJaxVector) = numpy.reshape(a, (-1,1))
Base.transpose(a::AbstractJaxMatrix) = numpy.transpose(a)

Base.adjoint(a::AbstractJaxArray{T}) where T<:Real = transpose(a)
Base.adjoint(a::AbstractJaxArray) = conj(transpose(a))

Base.PermutedDimsArray(a::AbstractJaxArray, perm) = permutedims(a, perm)
Base.permutedims(a::AbstractJaxArray) = transpose(a)
Base.permutedims(a::AbstractJaxArray{T,N}, perm::Dims{N}) where {T,N} =
    numpy.transpose(a, reverse(N.-perm))

Base.reshape(a::AbstractJaxArray, dims::Dims) = numpy.reshape(a, reverse(dims))
