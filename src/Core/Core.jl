module Core
    using ..PyUtils
    using LinearAlgebra

    export AbstractJaxArray, AbstractJaxVector, AbstractJaxMatrix
    export JaxAbstractArrayStyle
    export jl_to_np_type
    export JaxArray
    export jax, numpy, lax

    abstract type AbstractJaxArray{T,N} <: AbstractArray{T,N} end
    const AbstractJaxVector{T} = AbstractJaxArray{T,1}
    const AbstractJaxMatrix{T} = AbstractJaxArray{T,2}

    abstract type JaxAbstractArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

    const jax = PyNULL()
    const numpy = PyNULL()
    const lax = PyNULL()

    include("array.jl")
    include("traced.jl")
    include("batchtracer.jl")
    include("JVPTracer.jl")
    include("ParallelTracer.jl")

    include("indexing.jl")

    include("typeconversion.jl")

    function __init__()
        copy!(jax, pyimport_conda("jax", "jax"))
        copy!(numpy, pyimport_conda("jax.numpy", "jax"))
        copy!(lax, pyimport_conda("jax.lax", "jax"))

        # Automatic conversion to JaxArray for return of PyCall calls
        pytype_mapping(jax.interpreters.xla.DeviceArray, JaxArray)
        pytype_mapping(jax.interpreters.partial_eval.JaxprTracer, TracedArray)
        pytype_mapping(jax.interpreters.batching.BatchTracer, BatchTracerArray)
        pytype_mapping(jax.interpreters.ad.JVPTracer, JVPTracerArray)
        pytype_mapping(jax.interpreters.parallel.PapplyTracer, ParallelTracerArray)
    end
end
