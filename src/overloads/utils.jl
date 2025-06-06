#===============#
# Tracer unions #
#===============#

"""
    first_order_or(tracers)

Compute the most conservative elementwise OR of tracer sparsity patterns,
excluding second-order interactions of `HessianTracer`.

This is functionally equivalent to:
```julia
reduce(+, tracers)
```
"""
function first_order_or(ts::AbstractArray{T}) where {T <: AbstractTracer}
    # TODO: improve performance
    return reduce(first_order_or, ts; init = myempty(T))
end
function first_order_or(a::T, b::T) where {T <: GradientTracer}
    return gradient_tracer_2_to_1(a, b, false, false)
end
function first_order_or(a::T, b::T) where {T <: HessianTracer}
    return hessian_tracer_2_to_1(a, b, false, true, false, true, true)
end

"""
    second_order_or(tracers)

Compute the most conservative elementwise OR of tracer sparsity patterns,
including second-order interactions to update the `hessian` field of `HessianTracer`.

This is functionally equivalent to:
```julia
reduce(^, tracers)
```
"""
function second_order_or(ts::AbstractArray{T}) where {T <: AbstractTracer}
    # TODO: improve performance
    return reduce(second_order_or, ts; init = myempty(T))
end

function second_order_or(a::T, b::T) where {T <: GradientTracer}
    return gradient_tracer_2_to_1(a, b, false, false)
end
function second_order_or(a::T, b::T) where {T <: HessianTracer}
    return hessian_tracer_2_to_1(a, b, false, false, false, false, false)
end

#=================#
# Code generation #
#=================#

dims = (Symbol("1_to_1"), Symbol("2_to_1"), Symbol("1_to_2"))

# Generate both Gradient and Hessian code with one call to `generate_code_X_to_Y`
for d in dims
    f = Symbol("generate_code_", d)
    g = Symbol("generate_code_gradient_", d)
    h = Symbol("generate_code_hessian_", d)

    @eval function $f(M::Symbol, f::Function)
        expr_g = $g(M, f)
        expr_h = $h(M, f)
        return Expr(:block, expr_g, expr_h)
    end
end

# Allow all `generate_code_*` functions to be called on several operators at once
for d in dims
    for f in (
            Symbol("generate_code_", d),
            Symbol("generate_code_gradient_", d),
            Symbol("generate_code_hessian_", d),
        )
        @eval function $f(M::Symbol, ops::Union{AbstractVector, Tuple})
            exprs = [$f(M, op) for op in ops]
            return Expr(:block, exprs...)
        end
    end
end

# Overloads of 2-argument functions on arbitrary types
function generate_code_2_to_1_typed(M::Symbol, f::Function, Z::Type)
    expr_g = generate_code_gradient_2_to_1_typed(M, f, Z)
    expr_h = generate_code_hessian_2_to_1_typed(M, f, Z)
    return Expr(:block, expr_g, expr_h)
end
function generate_code_2_to_1_typed(M::Symbol, ops::Union{AbstractVector, Tuple}, Z::Type)
    exprs = [generate_code_2_to_1_typed(M, op, Z) for op in ops]
    return Expr(:block, exprs...)
end

## Overload operators
eval(generate_code_1_to_1(:Base, ops_1_to_1))
eval(generate_code_2_to_1(:Base, ops_2_to_1))
eval(generate_code_1_to_2(:Base, ops_1_to_2))

## List operators for later testing
test_operators_1_to_1(::Val{:Base}) = ops_1_to_1
test_operators_2_to_1(::Val{:Base}) = ops_2_to_1
test_operators_1_to_2(::Val{:Base}) = ops_1_to_2
