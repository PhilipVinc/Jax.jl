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

include("External/functors.jl")
include("External/nnlib.jl")

include("jit.jl")
include("vmap.jl")

function __init__()
    # Integration with other packages
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        include("External/flux.jl")
    end
end

# like gpu() from flux
tojax(m) = fmap(x -> adapt(JaxArray, x), m)

# adapt
struct ToArray end
Adapt.adapt_storage(::ToArray, xs::JaxArray) = convert(Array, xs)

Adapt.adapt_storage(::Type{<:JaxArray}, xs::Array) = JaxArray(xs)
Adapt.adapt_storage(::Type{<:Array}, xs::AbstractJaxArray) = convert(Array, xs)
convert_to_cpu(xs) = adapt(Array, xs)

Base.print_array(io::IO, X::JaxArray) =
    Base.print_array(io, adapt(ToArray(), X))

# show
Base._show_nonempty(io::IO, X::JaxArray, prefix::String) =
    Base._show_nonempty(io, adapt(ToArray(), X), prefix)
Base._show_empty(io::IO, X::JaxArray) =
    Base._show_empty(io, adapt(ToArray(), X))
Base.show_vector(io::IO, v::JaxArray, args...) =
    Base.show_vector(io, adapt(ToArray(), v), args...)

## collect to CPU (discarding wrapper type)

collect_to_cpu(xs::AbstractArray) = collect(adapt(ToArray(), xs))
#Base.collect(X::JaxArray) = collect_to_cpu(X)

end # module
