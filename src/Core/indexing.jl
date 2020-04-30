## Mostly taken from GPUArrays.jl


export allowscalar, @allowscalar, @disallowscalar, assertscalar

@enum ScalarIndexing ScalarAllowed ScalarWarned ScalarDisallowed

const scalar_allowed = Ref(ScalarDisallowed)
const scalar_warned = Ref(false)

"""
    allowscalar(allow=true, warn=true)
Configure whether scalar indexing is allowed depending on the value of `allow`.
If allowed, `warn` can be set to throw a single warning instead. Calling this function will
reset the state of the warning, and throw a new warning on subsequent scalar iteration.
"""
function allowscalar(allow::Bool=true, warn::Bool=true)
    scalar_warned[] = false
    scalar_allowed[] = if allow && !warn
        ScalarAllowed
    elseif allow
        ScalarWarned
    else
        ScalarDisallowed
    end
    return
end

"""
    assertscalar(op::String)
Assert that a certain operation `op` performs scalar indexing. If this is not allowed, an
error will be thrown ([`allowscalar`](@ref)).
"""
function assertscalar(op = "operation")
    if scalar_allowed[] == ScalarDisallowed
        error("$op is disallowed")
    elseif scalar_allowed[] == ScalarWarned && !scalar_warned[]
        @warn "Performing scalar operations on JAX arrays: This is very slow, consider disallowing these operations with `allowscalar(false)`"
        scalar_warned[] = true
    end
    return
end

"""
    @allowscalar ex...
    @disallowscalar ex...
    allowscalar(::Function, ...)
Temporarily allow or disallow scalar iteration.
Note that this functionality is intended for functionality that is known and allowed to use
scalar iteration (or not), i.e., there is no option to throw a warning. Only use this on
fine-grained expressions.
"""
macro allowscalar(ex)
    quote
        local prev = scalar_allowed[]
        scalar_allowed[] = ScalarAllowed
        local ret = $(esc(ex))
        scalar_allowed[] = prev
        ret
    end
end


#@doc (@doc @allowscalar) ->
macro disallowscalar(ex)
    quote
        local prev = scalar_allowed[]
        scalar_allowed[] = ScalarDisallowed
        local ret = $(esc(ex))
        scalar_allowed[] = prev
        ret
    end
end

function allowscalar(f::Base.Callable, allow::Bool=true, warn::Bool=false)
    prev = scalar_allowed[]
    allowscalar(allow, warn)
    ret = f()
    scalar_allowed[] = prev
    ret
end

# basic indexing

Base.IndexStyle(::Type{<:AbstractJaxArray}) = Base.IndexLinear()

function Base.getindex(xs::AbstractJaxArray{T}, i::Integer) where T
    assertscalar("scalar getindex")
    return collect(xs)[1]
end

#=function Base.setindex!(xs::AbstractJaxArray{T}, v::T, i::Integer) where T
    assertscalar("scalar setindex!")
    x = T[v]
    #copyto!(xs, i, x, 1, 1)
    error("not implemented")
    return xs
end=#

#Base.setindex!(xs::AbstractJaxArray, v, i::Integer) = xs[i] = convert(eltype(xs), v)
