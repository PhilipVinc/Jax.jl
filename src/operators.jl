import Base: +, -, *, /, //, %, &, |, ^, <<, >>, ⊻

# multiply
Base.:*(a::T, b::Number) where {T<:AbstractJaxArray} =
    convert(T, PyObject(b) * a.o)
Base.:*(a::Number, b::T) where {T<:AbstractJaxArray} =
    convert(T,  b.o * PyObject(a))
Base.:*(a::AbstractJaxMatrix, b::AbstractJaxMatrix) =
    jax.numpy.matmul( b.o, a.o)
Base.:*(a::AbstractJaxMatrix, b::AbstractJaxVector) =
    jax.numpy.dot( b.o, a.o)
Base.:*(a::AbstractJaxVector, b::AbstractJaxMatrix) =
    jax.numpy.dot( b.o, a.o)

const overridenbfuncs = :[+, -, *, /].args

# Julia base dispatches vec + vec to broadcast.
# But since we override the broadcast machinery to un-broadcast everything,
# this will result in an endless loop.
# So we override this and dispatch vec + vec to vec _+ vec which when broadcasted
# will call the correct jax function.
for op in overridenbfuncs
    isdefined(Base, op) || continue
    fname = Symbol("_", op)

    @eval begin
        $fname(a::T, b::T) where {T<:AbstractJaxArray} = convert(T, $op(a.o, b.o))
        $fname(a::T, b) where {T<:AbstractJaxArray} = convert(PyAny, $op(a.o, PyObject(b)))
        $fname(a, b::T) where {T<:AbstractJaxArray} = convert(PyAny, $op(PyObject(a), b.o))

        # If a tracer is applied to an JaxArray (a constant) then return a tracer.
        $fname(a::T, b::JaxArray) where {T<:AbstractJaxArray} = convert(T, $op(a.o, b.o))
        $fname(a::JaxArray, b::T) where {T<:AbstractJaxArray} = convert(T, $op(a.o, b.o))
        $fname(a::JaxArray, b::JaxArray) = convert(JaxArray, $op(a.o, b.o))

        $fname(a::AbstractJaxArray, b::AbstractJaxArray) = convert(PyAny, $op(a.o, b.o))

        jaxfunc(::typeof(Base.$op)) = $fname

        PyCall.pycall(::typeof($fname), ::Type, arg1, arg2) = $fname(arg1, arg2)
    end
end

#PyCall.pycall(::typeof(identity), ::Type, arg1) = convert(T, arg1)

@eval $(Symbol("_", :-))(a::T) where {T<:AbstractJaxArray} = convert(T, -(a.o))
@eval $(Symbol("_", :+))(a::T) where {T<:AbstractJaxArray} = convert(T, +(a.o))
