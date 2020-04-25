struct JaxArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

JaxArrayStyle(::Val{N}) where {N} = JaxArrayStyle{N}()
JaxArrayStyle{M}(::Val{N}) where {N,M} = JaxArrayStyle{N}()
Base.Broadcast.BroadcastStyle(::Type{<:JaxArray{T,N}}) where {T,N} =
    JaxArrayStyle{N}()

#Base.similar(bc::Broadcasted{CuArrayStyle{N}}, ::Type{T}) where {N,T} =
#    similar(CuArray{T}, axes(bc))

#Base.similar(bc::Broadcasted{CuArrayStyle{N}}, ::Type{T}, dims...) where {N,T} =
#    CuArray{T}(undef, dims...)

## replace base functions with libdevice alternatives

jaxfunc(f) = f
jaxfunc(::Type{T}) where {T} = (x...) -> T(x...) # broadcasting type ctors isn't GPU compatible

Base.Broadcast.broadcasted(::JaxArrayStyle{N}, f, args...) where {N} =
    Base.Broadcast.Broadcasted{JaxArrayStyle{N}}(jaxfunc(f), args, nothing)

Base.Broadcast.materialize(bc::Base.Broadcast.Broadcasted{<:JaxArrayStyle}) =
    convert(JaxArray, _materialize(bc))

_materialize(bc::Base.Broadcast.Broadcasted{<:JaxArrayStyle}) = begin
    args = map(_materialize, bc.args)
    bc.f(args...)
end

_materialize(bc) = bc


const libdevice =
    :[
        cos,
        cospi,
        sin,
        sinpi,
        tan,
        acos,
        asin,
        atan,
        cosh,
        sinh,
        tanh,
        acosh,
        asinh,
        atanh,
        log,
        log10,
        log1p,
        log2,
        logb,
        ilogb,
        exp,
        exp2,
        exp10,
        expm1,
        ldexp,
        erf,
        erfinv,
        erfc,
        erfcinv,
        erfcx,
    ].args

for f in libdevice
    isdefined(Base, f) || continue
    @eval jaxfunc(::typeof(Base.$f)) = np.$f
end

jaxliteral_pow(::typeof(^), x::T, ::Val{0}) where {T<:Real} = one(x)
jaxliteral_pow(::typeof(^), x::T, ::Val{1}) where {T<:Real} = x
jaxliteral_pow(::typeof(^), x::T, ::Val{2}) where {T<:Real} = x * x
jaxliteral_pow(::typeof(^), x::T, ::Val{3}) where {T<:Real} = x * x * x
jaxliteral_pow(::typeof(^), x::T, ::Val{p}) where {T<:Real,p} =
    np.power(x, Int32(p))

jaxfunc(::typeof(Base.literal_pow)) = jaxliteral_pow
jaxfunc(::typeof(Base.:(^))) = np.power

# test
using MacroTools

const _jaxfuncs = [copy(libdevice); :^]
jaxfuncs() = (global _jaxfuncs; _jaxfuncs)

_jaxint(x::Int) = Int32(x)#JaxArray{Int32,0}(Jax.np.int32(x), tuple())
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
                push!(new_x.args, :(one($sym)))
            else
                unroll = Expr(:call, :*)
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
        x = x in _jaxfuncs ? :(Jax.jaxfunc($x)) : x
        x = _jaxint(x)
        x = _jaxpowliteral(x)
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
