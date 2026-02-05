#= For now, three-argument functions are overloaded individually.
If this file grows too large:
    1. 3-arg operators should be classified in src/operators.jl
    2. the classification should be tested in test/classification.jl
    3. code generation utilities should be added to the src/overloads/*_tracer.jl files
=#
Base.clamp(t::T, lo, hi) where {T <: AbstractTracer} = t
Base.clamp(t::T, lo::T, hi) where {T <: AbstractTracer} = first_order_or(t, lo)
Base.clamp(t::T, lo, hi::T) where {T <: AbstractTracer} = first_order_or(t, hi)
function Base.clamp(t::T, lo::T, hi::T) where {T <: AbstractTracer}
    return first_order_or(t, first_order_or(lo, hi))
end

# For `fma(x, y, z)`, just fall back on `x * y + z`.
Base.fma(x::T, y::T, z::T) where {T <: Union{AbstractTracer, Dual}} = x * y + z
