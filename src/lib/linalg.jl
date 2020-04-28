using LinearAlgebra
LinearAlgebra.diag(a::AbstractJaxArray, k::Integer=0) = numpy.diag(a, -k)
