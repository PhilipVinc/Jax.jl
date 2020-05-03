module Jax

using Adapt
using Requires
using LinearAlgebra

export jax, np, JaxArray, JaxNumber
export block_until_ready
export tojax

export jit, grad, vmap

include("PyUtils/PyUtils.jl")
using .PyUtils

include("Core/Core.jl")
using .Core

include("Random/random.jl")
using .Random

include("TreeUtil/TreeUtil.jl")
using .TreeUtil


include("common.jl")

include("operators.jl")
include("jaxfuncs.jl")
include("broadcast.jl")

include("scalars.jl")

include("lib/_common.jl")
include("lib/array.jl")
include("lib/base.jl")
include("lib/linalg.jl")
include("lib/mapreduce.jl")
include("lib/statistics.jl")

include("External/nnlib.jl")

include("jit.jl")
include("vmap.jl")

function __init__()
    # Integration with other packages
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        include("External/flux.jl")
    end
end

# adapt
Adapt.adapt_storage(::Type{<:JaxArray}, xs::Array) = JaxArray(xs)
Adapt.adapt_storage(::Type{<:Array}, xs::AbstractJaxArray) = convert(Array, xs)
convert_to_cpu(xs) = adapt(Array, xs)

for (W, ctor) in (:AT => (A,mut)->mut(A), Adapt.wrappers...)
    @eval begin
        # display
        Base.print_array(io::IO, X::$W where {AT <: JaxArray}) =
            Base.print_array(io, $ctor(X, convert_to_cpu))

        # show
        Base._show_nonempty(io::IO, X::$W where {AT <: JaxArray}, prefix::String) =
            Base._show_nonempty(io, $ctor(X, convert_to_cpu), prefix)
        Base._show_empty(io::IO, X::$W where {AT <: JaxArray}) =
            Base._show_empty(io, $ctor(X, convert_to_cpu))
        Base.show_vector(io::IO, v::$W where {AT <: JaxArray}, args...) =
            Base.show_vector(io, $ctor(v, convert_to_cpu), args...)
    end
end

end # module
