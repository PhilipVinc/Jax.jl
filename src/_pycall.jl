# Equivalent to PyCall.PyArray but
# does not require the array to be writable, so that we can read back jax data
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
