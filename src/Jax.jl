module Jax

using PyCall
using Adapt

export jax, JaxArray

const jax = PyNULL()
const np = PyNULL()
const lax = PyNULL()
const random = PyNULL()

include("array.jl")
include("broadcast.jl")
include("operators.jl")
include("linalg.jl")

include("scalar_type_conversions.jl")

function __init__()
    copy!(jax, pyimport_conda("jax", "jax"))
    copy!(np, pyimport_conda("jax.numpy", "jax"))
    copy!(lax, pyimport_conda("jax.lax", "jax"))
    copy!(random, pyimport_conda("jax.random", "jax"))
end

# adapt
Adapt.adapt_storage(::Type{<:JaxArray}, xs::Array) = JaxArray(xs)

end # module
