function np_to_jl_type(o::PyObject)
    t = o.name
    if t == "float32"
        return Float32
    elseif t == "float64"
        return Float64
    elseif t == "complex64"
        return ComplexF32
    elseif t == "complex128"
        return Complexf64
    elseif t == "int32"
        return Int32
    elseif t == "int64"
        return Int64
    elseif t == "uint32"
        return UInt32
    elseif t == "uint64"
        return UInt64
    elseif t == "bool"
        return Bool
    else
        error("Unknown scalar type $t")
    end
end
