module Jax

using Adapt
using Requires
using LinearAlgebra

export jax, np, JaxArray
export block_until_ready
export tojax

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

include("lib/_common.jl")
include("lib/array.jl")
include("lib/base.jl")
include("lib/linalg.jl")
include("lib/mapreduce.jl")
include("lib/statistics.jl")

include("External/nnlib.jl")

function __init__()
    # Integration with other packages
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        include("External/flux.jl")
    end
end

# adapt
Adapt.adapt_storage(::Type{<:JaxArray}, xs::Array) = JaxArray(xs)
Adapt.adapt_storage(::Type{<:Array}, xs::AbstractJaxArray) = convert(Array, xs)

end # module
