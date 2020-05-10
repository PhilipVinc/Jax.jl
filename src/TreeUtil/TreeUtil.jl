module TreeUtil
    using ..PyUtils
    using ..Core

    export flatten, unflatten

    const _tree_util = PyNULL()

    function tree_flatten(tree)
        vals, f = _tree_util.tree_flatten(tree)
        vals = vals isa AbstractArray ? tuple(vals...) : vals
        return vals, f
    end

    function _flatten end
    function _unflatten end

    jl_flatten(x) = _flatten(x)
    jl_unflatten(f, x) = _unflatten(f, x)

    const _fwdfuncs = Dict(
        :Partial=>:Partial,
        :all_leaves=>:all_leaves,
        :build_tree=>:build_tree,
        :register_pytree_node=>:register_pytree_node,
        :tree_all=>:tree_all,
        :tree_leaves=>:tree_leaves,
        :tree_map=>:tree_map,
        :tree_multimap=>:tree_multimap,
        :tree_reduce=>:tree_reduce,
        :tree_structure=>:tree_structure,
        :tree_transpose=>:tree_transpose,
        :tree_unflatten=>:tree_unflatten,
        :treedef_children=>:treedef_children,
        :treedef_is_leaf=>:treedef_is_leaf,
        :treedef_tuple=>:treedef_tuple)
    for (jl,py) in _fwdfuncs
        @eval begin
            $jl(args...; kwargs...) = _tree_util.$py(args...;kwargs...)
        end
    end

    const flatten = tree_flatten
    const unflatten = tree_unflatten

    function __init__()
        copy!(_tree_util, jax.tree_util)


        # This is an horrinble hack.
        # it will register as a pytree node any wrapper of julia
        # objects.
        # It woudl be nice to internally dispatch on traits when calling
        TreeUtil.register_pytree_node(
            pytypeof(PyObject(identity)),
            jl_flatten,
            jl_unflatten,
        )
    end
end
