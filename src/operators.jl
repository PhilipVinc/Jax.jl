import Base: +, -, *, /, //, %, &, |, ^, <<, >>, ⊻

for op in (:+, :-, :*, :/, :%, :&, :|, :<<, :>>, :⊻)
    @eval begin
        $op(a::JaxArray, b::JaxArray) = convert(JaxArray, $op(a.o, b.o))
        $op(a::JaxArray, b) = convert(JaxArray, $op(a.o, PyObject(b)))
        $op(a, b::JaxArray) = convert(JaxArray, $op(PyObject(a), b.o))
        $op(a::JaxArray, b::Number) = convert(JaxArray, $op(a.o, PyObject(b)))
        $op(a::Number, b::JaxArray) = convert(JaxArray, $op(PyObject(a), b.o))
    end
end
