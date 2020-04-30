
for (f_jl, f_np) in libdevice_1
    isdefined(Base, f_jl) || continue
    @eval Base.$f_jl(a::AbstractJaxScalar{T}) where T = numpy.$f_np(a)
end

for (f_jl, f_np) in libdevice_2
    isdefined(Base, f_jl) || continue
    @eval Base.$f_jl(a::AbstractJaxScalar{T}, b::Number) where T = numpy.$f_np(a, b)
    @eval Base.$f_jl(a::Number, b::AbstractJaxScalar{T}) where T = numpy.$f_np(a, b)
    #@eval Base.$f_jl(a::AbstractJaxScalar{T}, b::AbstractJaxScalar{T2}) where {T,T2} = numpy.$f_np(a, b)
end

for op in overridenbfuncs
    bop = Symbol(".", op)
    @eval begin
        $op(a::AbstractJaxArray{T,0}, b) where T = $bop(a, b)
        $op(a, b::AbstractJaxArray{T,0}) where T = $bop(a, b)
        $op(a::AbstractJaxArray{T1,0},
                b::AbstractJaxArray{T2,0}) where {T1,T2} = $bop(a, b)


        $op(a::AbstractJaxArray{T,0}, b::Number) where T = $bop(a, b)
        $op(a::Number, b::AbstractJaxArray{T,0}) where T = $bop(a, b)
    end
end
