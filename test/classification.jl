using SparseConnectivityTracer:
    list_operators_1_to_1,
    is_der1_zero_global,
    is_der2_zero_global,
    is_der1_zero_local,
    is_der2_zero_local,
    list_operators_2_to_1,
    is_der1_arg1_zero_global,
    is_der2_arg1_zero_global,
    is_der1_arg2_zero_global,
    is_der2_arg2_zero_global,
    is_der_cross_zero_global,
    is_der1_arg1_zero_local,
    is_der2_arg1_zero_local,
    is_der1_arg2_zero_local,
    is_der2_arg2_zero_local,
    is_der_cross_zero_local,
    list_operators_1_to_2,
    is_der1_out1_zero_global,
    is_der2_out1_zero_global,
    is_der1_out2_zero_global,
    is_der2_out2_zero_global,
    is_der2_out1_zero_local,
    is_der1_out1_zero_local,
    is_der1_out2_zero_local,
    is_der2_out2_zero_local
using SpecialFunctions: SpecialFunctions
using NNlib: NNlib
using Test
using ForwardDiff: derivative, gradient, hessian

second_derivative(f, x) = derivative(_x -> derivative(f, _x), x)

DEFAULT_ATOL = 1e-8
DEFAULT_TRIALS = 20

## Random inputs

random_input(op) = rand()
random_input(::Union{typeof(acosh),typeof(acoth),typeof(acsc),typeof(asec)}) = 1 + rand()
random_input(::typeof(sincosd)) = 180 * rand()

random_first_input(op) = random_input(op)
random_second_input(op) = random_input(op)

## Skip tests on functions that don't support ForwardDiff's Dual numbers

correct_classification_1_to_1(op::typeof(!), x; atol) = true

## Derivatives and special cases

both_derivatives_1_to_1(op, x) = derivative(op, x), second_derivative(op, x)

function both_derivatives_1_to_1(::Union{typeof(big),typeof(widen)}, x)
    return both_derivatives_1_to_1(identity, x)
end
function both_derivatives_1_to_1(
    ::Union{typeof(floatmin),typeof(floatmax),typeof(maxintfloat)}, x
)
    return both_derivatives_1_to_1(zero, x)
end

function both_derivatives_2_to_1(op, x, y)
    return gradient(Base.splat(op), [x, y]), hessian(Base.splat(op), [x, y])
end

function both_derivatives_1_to_2(op, x)
    function op_vec(x)
        y = op(x)
        return [y[1], y[2]]
    end
    return derivative(op_vec, x), second_derivative(op_vec, x)
end

## 1-to-1

function correct_classification_1_to_1(op, x; atol)
    dfdx, d²fdx² = both_derivatives_1_to_1(op, x)
    if (is_der1_zero_global(op) | is_der1_zero_local(op, x)) && !isapprox(dfdx, 0; atol)
        return false
    elseif (is_der2_zero_global(op) | is_der2_zero_local(op, x)) &&
        !isapprox(d²fdx², 0; atol)
        return false
    else
        return true
    end
end

@testset verbose = true "1-to-1" begin
    @testset "$m" for m in (Base, SpecialFunctions, NNlib)
        @testset "$op" for op in list_operators_1_to_1(Val(Symbol(m)))
            @test all(
                correct_classification_1_to_1(op, random_input(op); atol=DEFAULT_ATOL) for
                _ in 1:DEFAULT_TRIALS
            )
            yield()
        end
    end
end;

## 2-to-1

function correct_classification_2_to_1(op, x, y; atol)
    g, H = both_derivatives_2_to_1(op, x, y)

    ∂f∂x    = g[1]
    ∂f∂y    = g[2]
    ∂²f∂x²  = H[1, 1]
    ∂²f∂y²  = H[2, 2]
    ∂²f∂x∂y = H[1, 2]

    if (is_der1_arg1_zero_global(op) | is_der1_arg1_zero_local(op, x, y)) &&
        !isapprox(∂f∂x, 0; atol)
        return false
    elseif (is_der2_arg1_zero_global(op) | is_der2_arg1_zero_local(op, x, y)) &&
        !isapprox(∂²f∂x², 0; atol)
        return false
    elseif (is_der1_arg2_zero_global(op) | is_der1_arg2_zero_local(op, x, y)) &&
        !isapprox(∂f∂y, 0; atol)
        return false
    elseif (is_der2_arg2_zero_global(op) | is_der2_arg2_zero_local(op, x, y)) &&
        !isapprox(∂²f∂y², 0; atol)
        return false
    elseif (is_der_cross_zero_global(op) | is_der_cross_zero_local(op, x, y)) &&
        !isapprox(∂²f∂x∂y, 0; atol)
        return false
    else
        return true
    end
end

@testset verbose = true "2-to-1" begin
    @testset "$m" for m in (Base, SpecialFunctions, NNlib)
        @testset "$op" for op in list_operators_2_to_1(Val(Symbol(m)))
            @test all(
                correct_classification_2_to_1(
                    op, random_first_input(op), random_second_input(op); atol=DEFAULT_ATOL
                ) for _ in 1:DEFAULT_TRIALS
            )
            yield()
        end
    end
end;

## 1-to-2

function correct_classification_1_to_2(op, x; atol)
    d1, d2 = both_derivatives_1_to_2(op, x)

    ∂f₁∂x = d1[1]
    ∂f₂∂x = d1[2]
    ∂²f₁∂x² = d2[1]
    ∂²f₂∂x² = d2[2]

    if (is_der1_out1_zero_global(op) | is_der1_out1_zero_local(op, x)) &&
        !isapprox(∂f₁∂x, 0; atol)
        return false
    elseif (is_der2_out1_zero_global(op) | is_der2_out1_zero_local(op, x)) &&
        !isapprox(∂²f₁∂x², 0; atol)
        return false
    elseif (is_der1_out2_zero_global(op) | is_der1_out2_zero_local(op, x)) &&
        !isapprox(∂f₂∂x, 0; atol)
        return false
    elseif (is_der2_out2_zero_global(op) | is_der2_out2_zero_local(op, x)) &&
        !isapprox(∂²f₂∂x², 0; atol)
        return false
    else
        return true
    end
end

@testset verbose = true "1-to-2" begin
    @testset "$m" for m in (Base, SpecialFunctions, NNlib)
        @testset "$op" for op in list_operators_1_to_2(Val(Symbol(m)))
            @test all(
                correct_classification_1_to_2(op, random_input(op); atol=DEFAULT_ATOL) for
                _ in 1:DEFAULT_TRIALS
            )
            yield()
        end
    end
end;
