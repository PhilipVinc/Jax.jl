#=struct VmapFunc
    fun
    in_axes
    out_axes
    obj
end

_ndims(x::AbstractArray) = ndims(x)
_ndims(x::AbstractArray{<:AbstractArray}) = map(ndims, x)

_max_ndims(x::AbstractArray{<:Number}) = ndims(x)
_max_ndims(x::AbstractArray) = maximum(map(ndims, x))
_max_ndims(x::Union{Tuple,NamedTuple,Dict})

mapover(f, iselement, x) = iselement(x) ? f(x) : map(e -> mapover(f, iselement, e), x)
mapover(f, iselement, x::Union{Dict, NamedTuple}) = iselement(x) ? f(x) : map(e -> mapover(f, iselement, e), values(x))

mapreduceover(f, op, iselement, x) = iselement(x) ? f(x) : reduce(op, map(e -> mapreduceover(f, op, iselement, e), x))
mapreduceover(f, op, iselement, x::Union{Dict, NamedTuple}) = iselement(x) ? f(x) : reduce(op, map(e -> mapreduceover(f, op, iselement, e), values(x)))

_isleaf(x) = x isa Number || x isa AbstractArray{<:Number}

(f::Vmapfunc)(args...; kwargs...)=begin
    isnothing(f.obj) || f.obj(args...; kwargs...)
    in_axes = f.in_axes
    out_axes = f.out_axes

    if in_axes isa Colon
        py_in_axes = 0
    elseif in_axes isa Integer
        maxdim = mapreduceover(ndims, max, _isleaf, args)
        py_in_axes = maxdim - in_axes
    elseif in_axes isa Tuple

    end
    if f._in
end=#

"""
    vmap(f, in_axes, out_axes)

Vectorizing map. Creates a function which maps fun over argument axes.

Parameters:
    fun (Callable) – Function to be mapped over additional axes.
    in_axes – A nonnegative integer, None, or (nested) standard Python container
        (tuple/list/dict) thereof specifying which input array axes to map over.
        If each positional argument to fun is an array, then in_axes can be a nonnegative
        integer, a None, or a tuple of integers and Nones with length equal to the number
        of positional arguments to fun. An integer or None indicates which array
        axis to map over for all arguments (with None indicating not to map any axis),
        and a tuple indicates which axis to map for each corresponding positional argument.
        If the positional arguments to fun are container types, the corresponding element
        of in_axes can itself be a matching container, so that distinct array axes can be
        mapped for different container elements. in_axes must be a container tree prefix of
        the positional argument tuple passed to fun.
        At least one positional argument must have in_axes not None. The sizes of the
        mapped input axes for all mapped positional arguments must all be equal.

    out_axes – A nonnegative integer, None, or (nested) standard Python container
        (tuple/list/dict) thereof indicating where the mapped axis should appear in
        the output. All outputs with a mapped axis must have a non-None out_axes specification.
Return type:
        Callable

Returns:
    Batched/vectorized version of fun with arguments that correspond to those of fun,
    but with extra array axes at positions indicated by in_axes, and a return value
    that corresponds to that of fun, but with extra array axes at positions indicated by out_axes.
"""
#vmap(f; in_axes, out_axes=:) = vmap(f, in_axes, out_axes=out_axes)
vmap(f; in_axes_fromend, out_axes_fromend=0) = vmap_endorder(f, in_axes_fromend, out_axes=out_axes)

function vmap(f, in_axes; out_axes=:)
    throw(ErrorException("Please use the argument in_axes_fromend, and write the batched dimensions
    starting from the last (0), second last (1)... etc.
    This is required because arrays are stored transposed."))
end

function vmap_endorder(f, in_axes=0, out_axes=0)
    # Change from Julia's 1 based to Python's 0 based indexing
    return jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
end
