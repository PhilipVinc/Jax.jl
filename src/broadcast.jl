using Base: Broadcast

Base.broadcasted(::JAS, f, args...) where {JAS<:JaxAbstractArrayStyle} =
    Broadcast.Broadcasted{JAS}(jaxfunc(f), args, nothing)

_materialize(bc::Broadcast.Broadcasted{<:JaxAbstractArrayStyle}) = begin
    args = map(_materialize, bc.args)
    # this is most likely a pycall returning PyObject.
    # We convert it back only in the outermost layer
    # this could be simplified if we used only one
    # bvroadcastarraystyle and used pycall(PyAny) here.
    bc.f(args...)
end

_materialize(bc) = bc
