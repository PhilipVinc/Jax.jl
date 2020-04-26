import Base: +, -, *, /, //, %, &, |, ^, <<, >>, ⊻

# multiply
Base.:*(a::JaxArray, b::Number) = convert(JaxArray, a.o * PyObject(b))
Base.:*(a::Number, b::JaxArray) = convert(JaxArray, PyObject(a) * b.o)
Base.:*(a::JaxMatrix, b::JaxMatrix) =
    convert(JaxArray, jax.numpy.matmul(a.o, b.o))
Base.:*(a::JaxMatrix, b::JaxVector) =
    convert(JaxArray, jax.numpy.matmul(a.o, b.o))
Base.:*(a::JaxVector, b::JaxMatrix) =
    convert(JaxArray, jax.numpy.matmul(a.o, b.o))

const overridenbfuncs = :[+, -, *, /].args

for op in overridenbfuncs
    isdefined(Base, op) || continue
    fname = Symbol("_", op)

    @eval begin
        $fname(a::JaxArray, b::JaxArray) = convert(JaxArray, $op(a.o, b.o))
        $fname(a::JaxArray, b) = convert(JaxArray, $op(a.o, PyObject(b)))
        $fname(a, b::JaxArray) = convert(JaxArray, $op(PyObject(a), b.o))

        jaxfunc(::typeof(Base.$op)) = $fname
    end
end

@eval $(Symbol("_", :-))(a::JaxArray) = convert(JaxArray, -(a.o))
@eval $(Symbol("_", :+))(a::JaxArray) = convert(JaxArray, +(a.o))
