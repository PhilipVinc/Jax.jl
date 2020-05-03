module Random

using Random
using PyCall
using ..Core

const _random = PyNULL()

include("PRNG.jl")
include("distributions.jl")
include("wrappers.jl")

function __init__()
  copy!(_random, pyimport("jax.random"))
end

end
