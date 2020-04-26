module Jax

using PyCall
using Adapt
using Requires

export jax, JaxArray

const jax = PyNULL()
const np = PyNULL()
const lax = PyNULL()
const random = PyNULL()

include("array.jl")
include("operators.jl")
include("broadcast.jl")
include("linalg.jl")

include("scalar_type_conversions.jl")

#random
include("random.jl")

include("_pycall.jl")

function __init__()
    copy!(jax, pyimport_conda("jax", "jax"))
    copy!(np, pyimport_conda("jax.numpy", "jax"))
    copy!(lax, pyimport_conda("jax.lax", "jax"))
    copy!(random, pyimport_conda("jax.random", "jax"))

    # Automatic conversion to JaxArray for return of PyCall calls
    pytype_mapping(jax.interpreters.xla.DeviceArray, JaxArray)

    # Integration with other packages
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        include("flux.jl")
    end
end

# adapt
Adapt.adapt_storage(::Type{<:JaxArray}, xs::Array) = JaxArray(xs)

end # module
