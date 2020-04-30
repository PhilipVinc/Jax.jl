using LinearAlgebra

LinearAlgebra.diag(a::AbstractJaxArray, k::Integer=0) = numpy.diag(a, -k)

LinearAlgebra.kron(a::AbstractJaxArray, b::AbstractJaxArray) = numpy.kron(a, b)

LinearAlgebra.tr(a::AbstractJaxMatrix, diag=0) = numpy.trace(a, offset=-diag)

LinearAlgebra.dot(a::AbstractJaxVector, b::AbstractJaxVector) = numpy.dot(a, b)
