## replace base functions with libdevice alternatives

jaxfunc(f) = f
jaxfunc(::Type{T}) where {T} = (x...) -> T(x...) # broadcasting type ctors isn't GPU compatible

const libdevice_1 = Dict(
#complex
:abs=>:abs,
:angle=>:angle,
:conj=>:conj,
:real=>:real, :imag=>:imag,
#trig
:cos=>:cos, :cosh=>:cosh,
:acos=>:arccos, :acosh=>:arccosh,
:sin=>:sin, :sinh=>:sinh,
:asin=>:arcsin, :asinh=>:arcsinh,
:tan=>:tan, :tanh=>:tanh,
:atan=>:arctan, :atanh=>:arctanh,
#
:ceil=>:ceil, :floor=>:floor,
#math
:exp=>:exp, :expm1=>:expm1, :exp2=>:exp2,
:log=>:log, :log1p=>:log1p, :log10=>:log10, :log2=>:log2,
:sqrt=>:sqrt, :inv=>:reciprocal)

const libdevice_2 = Dict(
#comparison
:(==)=>:equal, :!= => :not_equal,
:> => :greater, :>= => :greater_equal,
:< => :less, :<= => :less_equal,
)

#const libdevice_3 = Dict(
#
#)

const libdevice = merge(libdevice_1,
                        libdevice_2,
                        #libdevice_3
                        )

for (f_jl, f_np) in libdevice
    isdefined(Base, f_jl) || continue
    @eval jaxfunc(::typeof(Base.$f_jl)) = numpy.$f_np
    #@eval Base.$f(x::AbstractJaxArray) = $f(x)
end

jaxliteral_pow(::typeof(^), x::AbstractJaxArray, ::Val{0}) = one(x)
jaxliteral_pow(::typeof(^), x::AbstractJaxArray, ::Val{1}) = x
jaxliteral_pow(::typeof(^), x::AbstractJaxArray, ::Val{2}) = x * x
jaxliteral_pow(::typeof(^), x::AbstractJaxArray, ::Val{3}) = x * x * x
jaxliteral_pow(::typeof(^), x::AbstractJaxArray, ::Val{p}) where p = numpy.power(x, Int32(p))

# ??
jaxliteral_pow(::Base.RefValue{typeof(^)}, x::AbstractJaxArray, ::Base.RefValue{Val{p}}) where p = numpy.power(x, Int32(p))

jaxfunc(::typeof(Base.literal_pow)) = jaxliteral_pow
jaxfunc(::typeof(Base.:(^))) = power

# test
using MacroTools

const _jaxfuncs = [keys(libdevice); :^; copy(overridenbfuncs)]
jaxfuncs() = (global _jaxfuncs; _jaxfuncs)

_jaxint(x::Int) = Int32(x)#JaxArray{Int32,0}(Jax.int32(x), tuple())
_jaxint(x::Expr) = x.head == :call && x.args[1] == :Int32 && x.args[2] isa Int ?
    Int32(x.args[2]) : x
_jaxint(x) = x

function _jaxpowliteral(x::Expr)
    if x.head == :call && x.args[1] == :(Jax.jaxfunc(^)) && x.args[3] isa Int32
        num = x.args[3]
        if 0 <= num <= 3
            sym = gensym(:x)
            new_x = Expr(:block, :($sym = $(x.args[2])))

            if iszero(num)
                #push!(new_x.args, :(one($sym)))
                push!(new_x.args, :(one(eltype($sym))))
            elseif isone(num)
                push!(new_x.args, sym)
            else
                unroll = Expr(:call, :(Jax.jaxfunc(*)))
                for x = one(num):num
                    push!(unroll.args, sym)
                end
                push!(new_x.args, unroll)
            end

            x = new_x
        end
    end
    x
end
_jaxpowliteral(x) = x

function replace_device(ex)
    global _jaxfuncs
    MacroTools.postwalk(ex) do x
        x = _jaxpowliteral(x)
        x = x in _jaxfuncs ? :(Jax.jaxfunc($x)) : x
        x = _jaxint(x)
        x
    end
end

macro jaxfunc(ex)
    global _jaxfuncs
    def = MacroTools.splitdef(ex)
    f = def[:name]
    def[:name] = Symbol(:jax, f)
    def[:body] = replace_device(def[:body])
    push!(_jaxfuncs, f)
    quote
        $(esc(MacroTools.combinedef(def)))
        Jax.jaxfunc(::typeof($(esc(f)))) = $(esc(def[:name]))
    end
end
