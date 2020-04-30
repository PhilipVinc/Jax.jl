abstract type AbstractJaxTracer{T,N} <: AbstractJaxArray{T,N} end
#  o::PyObject
#  dims::NTuple{N,Int}
#end

const AbstractTracerVector{T} = AbstractJaxTracer{T,1}
const AbstractTracerMatrix{T} = AbstractJaxTracer{T,2}

Base.size(a::AbstractJaxTracer) = a.dims
Base.length(a::AbstractJaxTracer) = prod(a.dims)
Base.eltype(a::AbstractJaxTracer{T}) where {T} = T
Base.axes(a::AbstractJaxTracer) = map(Base.OneTo, size(a))

#wrap python properties
aval(a::AbstractJaxTracer) = a.o.aval

PyCall.PyObject(a::AbstractJaxTracer) = a.o

function Base.show(io::IO, ::MIME"text/plain", a::TA) where {TA<:AbstractJaxTracer}
  println(io, "$(_szstr(a.dims)) $TA:")
  print(io, " ", a.o)
end

function Base.show(io::IO, a::TA) where {TA<:AbstractJaxTracer}
  print(io, "$(_szstr(a.dims)) $TA")
end

for name in (:TracedArray, :ParallelTracerArray, :JVPTracerArray, :BatchTracerArray)
  @eval begin
    struct $name{T,N} <: AbstractJaxTracer{T,N}
      o::PyObject
      dims::NTuple{N,Int}
    end

    function $name(parr::PyObject)
      aval = parr.aval
      T = np_to_jl_type(aval.dtype)
      return $name{T,length(aval.shape)}(parr, reverse(aval.shape))
    end
    Base.convert(::Type{<:$name}, o::PyObject) = $name(o)
  end
end
