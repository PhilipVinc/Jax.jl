module Jax

using PyCall
using Adapt
using Requires

export jax, np, JaxArray

const jax = PyNULL()
const np = PyNULL()
const lax = PyNULL()
const random = PyNULL()

abstract type AbstractJaxArray{T,N} <: AbstractArray{T,N} end
const AbstractJaxVector{T} = AbstractJaxArray{T,1}
const AbstractJaxMatrix{T} = AbstractJaxArray{T,2}

abstract type JaxAbstractArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

include("Array/array.jl")
include("Array/broadcast.jl")

include("Traced/traced.jl")
include("Traced/broadcast.jl")

include("Batching/batchtracer.jl")
include("Batching/broadcast.jl")

include("AD/JVPTracer.jl")
include("AD/broadcast.jl")

include("Parallel/ParallelTracer.jl")
include("Parallel/broadcast.jl")

include("operators.jl")
include("jaxfuncs.jl")
include("broadcast.jl")
include("random.jl")

include("scalar_type_conversions.jl")

#upstream fixes
include("_pycall.jl")

function __init__()
    copy!(jax, pyimport_conda("jax", "jax"))
    copy!(np, pyimport_conda("jax.numpy", "jax"))
    copy!(lax, pyimport_conda("jax.lax", "jax"))
    copy!(random, pyimport_conda("jax.random", "jax"))

    # Automatic conversion to JaxArray for return of PyCall calls
    pytype_mapping(jax.interpreters.xla.DeviceArray, JaxArray)
    pytype_mapping(jax.interpreters.partial_eval.JaxprTracer, TracedArray)
    pytype_mapping(jax.interpreters.batching.BatchTracer, BatchTracerArray)
    pytype_mapping(jax.interpreters.ad.JVPTracer, JVPTracerArray)
    pytype_mapping(jax.interpreters.parallel.PapplyTracer, ParallelTracerArray)

    # Integration with other packages
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        include("flux.jl")
    end
end

# adapt
Adapt.adapt_storage(::Type{<:JaxArray}, xs::Array) = JaxArray(xs)

end # module
