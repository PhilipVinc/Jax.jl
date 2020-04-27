module PyUtils
    using Reexport
    @reexport using PyCall

    export jl_to_np_type, np_to_jl_type
    export PyArray_ReadOnly

    const _pyslice = PyNULL()
    const _pycolon = PyNULL()

    # Upstream fixes and new conversions
    include("PyCall.jl")

    include("scalar_type_conversions.jl")

    function __init__()
        copy!(_pyslice, py"slice")
        copy!(_pycolon, _pyslice(nothing, nothing, nothing))
    end
end
