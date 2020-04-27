# Equivalent to PyCall.PyArray but
# does not require the array to be writable, so that we can read back jax data
function PyCall.PyArray(o::PyObject, readonly::Bool)
    if !readonly
        return PyCall.PyArray(o)
    else
        return PyArray_ReadOnly(o)
    end
end

function PyArray_ReadOnly(o::PyObject)
    info = PyArray_Info_ReadOnly(o)
    return PyCall.PyArray{eltype(info),length(info.sz)}(o, info)
end

function PyArray_Info_ReadOnly(o::PyObject)
    # n.b. the pydecref(::PyBuffer) finalizer handles releasing the PyBuffer
    pybuf = PyCall.PyBuffer(o, PyCall.PyBUF_ND_STRIDED & ~PyCall.PyBUF_WRITABLE)
    T, native_byteorder = PyCall.array_format(pybuf)
    sz = size(pybuf)
    strd = strides(pybuf)
    length(strd) == 0 && (sz = ())
    N = length(sz)
    isreadonly = pybuf.buf.readonly == 1
    return PyCall.PyArray_Info{T,N}(
        native_byteorder,
        sz,
        strd,
        pybuf.buf.buf,
        isreadonly,
        pybuf,
    )
end

PyCall.getindex(o::PyObject, i::PyObject) = PyCall.get(o, i)
PyCall.getindex(o::PyObject, ::Colon) = PyCall.get(o, :)

PyCall.PyObject(::Colon) = _pycolon

##
#=
const _pynamedtuples = Dict{Any, Any}()

struct PyStructSequenceField2
    name::Cstring
    doc::Cstring
end

struct PyStructSequenceFieldJl
    name::String
    doc::String
end
Base.cconvert(::Type{PyStructSequenceField2}, v::PyStructSequenceFieldJl) =
    PyStructSequenceField2(Base.cconvert(Cstring, v.name), Base.cconvert(Cstring, v.doc))

Base.unsafe_convert(::Type{PyStructSequenceField2}, v::Tuple) = PyStructSequenceField2(Base.unsafe_convert(Cstring, v[1]), Base.unsafe_convert(Cstring, v[2]))



struct PyStructSequenceDesc
    name::Cstring
    doc::Cstring
    fields::Vector{Ptr{PyStructSequenceField}}
    n_in_sequence::Cint
end

function PyStructSequenceDesc(T::Type{<:NamedTuple})
    name = "namedtuple_$(length(_pynamedtuples))"
    nm = Base.unsafe_convert(Cstring, name)
    names = [string(nm) for nm=T.names]
    Cnames = [Base.unsafe_convert(Cstring, nm) for nm=names]
    fields = [PyStructSequenceField(nm, C_NULL) for nm=Cnames]
    flds = [convert(Ptr{PyStructSequenceField},Base.pointer_from_objref(f)) for f in fields]
    push!(flds, C_NULL)
    return PyStructSequenceDesc(nm,
                                C_NULL,
                                flds,
                                length(fields))
end

function PyCall.prepycheck(o::PyObject)


    return false
end

function PyCall.PyObject(t::NamedTuple)
    T = typeof(t)

    if T ∈ keys(_pynamedtuples)
        pyT = _pynamedtuples[T]
    else
        pyT = PyStructSequenceDesc(T)

        oT = PyObject(PyCall.@pycheckn ccall((PyCall.@pysym :PyStructSequence_NewType), PyPtr, (Ptr{PyStructSequenceDesc},), Ref(pyT)))
        _pynamedtuples[T] = oT
    end

    #=len = lastindex(t) # lastindex, not length, because of julia#14924
    o = PyObject(PyCall.@pycheckn ccall((PyCall.@pysym :PyTuple_New), PyPtr, (Int,), len))
    for i = 1:len
        oi = PyObject(t[i])
        PyCall.@pycheckz ccall((PyCall.@pysym :PyTuple_SetItem), Cint, (PyPtr,Int,PyPtr),
                         o, i-1, oi)
        PyCall.pyincref(oi) # PyTuple_SetItem steals the reference
    end
    return o=#
end
=#
