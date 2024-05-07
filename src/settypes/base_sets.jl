function keys2set(::Type{S}, d::Dict{I}) where {I<:Integer,S<:AbstractSet{<:I}}
    return S(keys(d))
end

# Performance can be gained by not re-allocating empty tracers
## BitSet
const EMPTY_CONNECTIVITY_TRACER_BITSET = ConnectivityTracer(BitSet())
const EMPTY_JACOBIAN_TRACER_BITSET     = JacobianTracer(BitSet())
const EMPTY_HESSIAN_TRACER_BITSET      = HessianTracer(Dict{Int,BitSet}())

empty(::Type{ConnectivityTracer{Int,BitSet}})             = EMPTY_CONNECTIVITY_TRACER_BITSET
empty(::Type{JacobianTracer{Int,BitSet}})                 = EMPTY_JACOBIAN_TRACER_BITSET
empty(::Type{HessianTracer{Int,BitSet,Dict{Int,BitSet}}}) = EMPTY_HESSIAN_TRACER_BITSET

## Set
const EMPTY_CONNECTIVITY_TRACER_SET_U8  = ConnectivityTracer(Set{UInt8}())
const EMPTY_CONNECTIVITY_TRACER_SET_U16 = ConnectivityTracer(Set{UInt16}())
const EMPTY_CONNECTIVITY_TRACER_SET_U32 = ConnectivityTracer(Set{UInt32}())
const EMPTY_CONNECTIVITY_TRACER_SET_U64 = ConnectivityTracer(Set{UInt64}())

const EMPTY_JACOBIAN_TRACER_SET_U8  = JacobianTracer(Set{UInt8}())
const EMPTY_JACOBIAN_TRACER_SET_U16 = JacobianTracer(Set{UInt16}())
const EMPTY_JACOBIAN_TRACER_SET_U32 = JacobianTracer(Set{UInt32}())
const EMPTY_JACOBIAN_TRACER_SET_U64 = JacobianTracer(Set{UInt64}())

const EMPTY_HESSIAN_TRACER_SET_U8  = HessianTracer(Dict{UInt8,Set{UInt8}}())
const EMPTY_HESSIAN_TRACER_SET_U16 = HessianTracer(Dict{UInt16,Set{UInt16}}())
const EMPTY_HESSIAN_TRACER_SET_U32 = HessianTracer(Dict{UInt32,Set{UInt32}}())
const EMPTY_HESSIAN_TRACER_SET_U64 = HessianTracer(Dict{UInt64,Set{UInt64}}())

empty(::Type{ConnectivityTracer{UInt8,Set{UInt8}}})   = EMPTY_CONNECTIVITY_TRACER_SET_U8
empty(::Type{ConnectivityTracer{UInt16,Set{UInt16}}}) = EMPTY_CONNECTIVITY_TRACER_SET_U16
empty(::Type{ConnectivityTracer{UInt32,Set{UInt32}}}) = EMPTY_CONNECTIVITY_TRACER_SET_U32
empty(::Type{ConnectivityTracer{UInt64,Set{UInt64}}}) = EMPTY_CONNECTIVITY_TRACER_SET_U64

empty(::Type{JacobianTracer{UInt8,Set{UInt8}}})   = EMPTY_JACOBIAN_TRACER_SET_U8
empty(::Type{JacobianTracer{UInt16,Set{UInt16}}}) = EMPTY_JACOBIAN_TRACER_SET_U16
empty(::Type{JacobianTracer{UInt32,Set{UInt32}}}) = EMPTY_JACOBIAN_TRACER_SET_U32
empty(::Type{JacobianTracer{UInt64,Set{UInt64}}}) = EMPTY_JACOBIAN_TRACER_SET_U64

empty(::Type{HessianTracer{UInt8,Set{UInt8},Dict{UInt8,Set{UInt8}}}})     = EMPTY_HESSIAN_TRACER_SET_U8
empty(::Type{HessianTracer{UInt16,Set{UInt16},Dict{UInt16,Set{UInt16}}}}) = EMPTY_HESSIAN_TRACER_SET_U16
empty(::Type{HessianTracer{UInt32,Set{UInt32},Dict{UInt32,Set{UInt32}}}}) = EMPTY_HESSIAN_TRACER_SET_U32
empty(::Type{HessianTracer{UInt64,Set{UInt64},Dict{UInt64,Set{UInt64}}}}) = EMPTY_HESSIAN_TRACER_SET_U64
