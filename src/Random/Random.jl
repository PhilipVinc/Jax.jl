module Random

using Random
using PyCall
using ..Core

const _random = PyNULL()

include("PRNG.jl")
include("distributions.jl")
include("wrappers.jl")

function __init__()
  copy!(_random, pyimport_conda("jax.random", "jax"))
end

end
