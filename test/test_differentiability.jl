using ForwardDiff: derivative, gradient, hessian
using SparseConnectivityTracer:
    ops_1_to_1,
    ops_1_to_1_s,
    ops_1_to_1_f,
    ops_1_to_1_z,
    ops_1_to_1_const,
    ops_2_to_1,
    ops_2_to_1_ssc,
    ops_2_to_1_ssz,
    ops_2_to_1_sfc,
    ops_2_to_1_sfz,
    ops_2_to_1_fsc,
    ops_2_to_1_fsz,
    ops_2_to_1_ffc,
    ops_2_to_1_ffz,
    ops_2_to_1_szz,
    ops_2_to_1_zsz,
    ops_2_to_1_fzz,
    ops_2_to_1_zfz,
    ops_2_to_1_zzz,
    ops_1_to_2,
    ops_1_to_2_ss,
    ops_1_to_2_sf,
    ops_1_to_2_fs,
    ops_1_to_2_ff,
    ops_1_to_2_sz,
    ops_1_to_2_zs,
    ops_1_to_2_fz,
    ops_1_to_2_zf,
    ops_1_to_2_zz
using Test

DEFAULT_ATOL = 1e-8
isapproxzero(x; atol=DEFAULT_ATOL) = abs(x) <= atol

random_input(f) = rand()
random_input(::Union{typeof(acosh),typeof(acoth),typeof(acsc),typeof(asec)}) = 1 + rand()

random_first_input(f) = random_input(f)
random_second_input(f) = random_input(f)

second_derivative(f, x) = derivative(_x -> derivative(f, _x), x)

sym2fn(op::Symbol) = @eval Base.$op

# Use enum as return type
@enum Order begin
    zero_order   = 0
    first_order  = 1
    second_order = 2
    error_order  = -1
end

function differentiability(∂f∂x, ∂²f∂x²; atol=DEFAULT_ATOL)
    is_∂f∂x_zero   = isapproxzero(∂f∂x; atol)
    is_∂²f∂x²_zero = isapproxzero(∂²f∂x²; atol)

    if !is_∂f∂x_zero
        if !is_∂²f∂x²_zero
            return second_order
        else
            return first_order
        end
    else # is_∂f∂x_zero
        if is_∂²f∂x²_zero
            return zero_order
        else
            return error_order
        end
    end
end

## 1-to-1

function classify_1_to_1(f, x; atol=DEFAULT_ATOL)
    ∂f∂x = derivative(f, x)
    ∂²f∂x² = second_derivative(f, x)

    order = differentiability(∂f∂x, ∂²f∂x²; atol)
    order == error_order && @warn "Weird behavior" f x ∂f∂x ∂²f∂x²
    return order
end

function classify_1_to_1(op::Symbol; atol=DEFAULT_ATOL, trials=100)
    f = sym2fn(op)
    try
        return maximum(classify_1_to_1(f, random_input(f); atol) for _ in 1:trials)
    catch e
        @warn "Classification of 1-to-1 operator $op failed" e
        return error_order
    end
end

const TEST_1_TO_1 = (
    ("Second order", ops_1_to_1_s, second_order),
    ("First order", ops_1_to_1_f, first_order),
    ("Zero order", ops_1_to_1_z, zero_order),
    ("Constant", ops_1_to_1_const, zero_order),
)
@testset verbose = true "1-to-1" begin
    @testset "All operators covered" begin
        all_ops = union([ops for (name, ops, ref_order) in TEST_1_TO_1]...)
        @test Set(all_ops) == Set(ops_1_to_1)
    end
    for (name, ops, ref_order) in TEST_1_TO_1
        @testset "$name" begin
            for op in ops
                @testset "$op" begin
                    @test classify_1_to_1(op) == ref_order
                end
            end
        end
    end
end;

## 2-to-1

function classify_2_to_1(f, x, y; atol)
    g = gradient(Base.splat(f), [x, y])
    H = hessian(Base.splat(f), [x, y])

    ∂f∂x    = g[1]
    ∂f∂y    = g[2]
    ∂²f∂x²  = H[1, 1]
    ∂²f∂y²  = H[2, 2]
    ∂²f∂x∂y = H[1, 2]

    first_arg = differentiability(∂f∂x, ∂²f∂x²; atol)
    first_arg == error_order && @warn "Weird behavior on argument x" f x y ∂f∂x ∂²f∂x²

    second_arg = differentiability(∂f∂y, ∂²f∂y²; atol)
    second_arg == error_order && @warn "Weird behavior on argument y" f x y ∂f∂y ∂²f∂y²

    cross = isapproxzero(∂²f∂x∂y; atol) ? zero_order : second_order
    return (first_arg, second_arg, cross)
end

classify_2_to_1(op::Symbol; kwargs...) = classify_2_to_1(sym2fn(op); kwargs...)
function classify_2_to_1(op::Symbol; atol=1e-5, trials=100)
    f = sym2fn(op)
    try
        return maximum(
            classify_2_to_1(f, random_first_input(f), random_second_input(f); atol) for
            _ in 1:trials
        )
    catch e
        @warn "Classification of 2-to-1 operator `$op` failed" e
        return (error_order, error_order, error_order)
    end
end

const TEST_2_TO_1 = (
    ("ssc", ops_2_to_1_ssc, (second_order, second_order, second_order)),
    ("ssz", ops_2_to_1_ssz, (second_order, second_order, zero_order)),
    ("sfc", ops_2_to_1_sfc, (second_order, first_order, second_order)),
    ("sfz", ops_2_to_1_sfz, (second_order, first_order, zero_order)),
    ("fsc", ops_2_to_1_fsc, (first_order, second_order, second_order)),
    ("fsz", ops_2_to_1_fsz, (first_order, second_order, zero_order)),
    ("ffc", ops_2_to_1_ffc, (first_order, first_order, second_order)),
    ("ffz", ops_2_to_1_ffz, (first_order, first_order, zero_order)),
    ("szz", ops_2_to_1_szz, (second_order, zero_order, zero_order)),
    ("zsz", ops_2_to_1_zsz, (zero_order, second_order, zero_order)),
    ("fzz", ops_2_to_1_fzz, (first_order, zero_order, zero_order)),
    ("zfz", ops_2_to_1_zfz, (zero_order, second_order, zero_order)),
    ("zzz", ops_2_to_1_zzz, (zero_order, zero_order, zero_order)),
)
@testset verbose = true "2-to-1" begin
    @testset "All operators covered" begin
        all_ops = union([ops for (name, ops, ref_order) in TEST_2_TO_1]...)
        @test Set(all_ops) == Set(ops_2_to_1)
    end
    for (name, ops, ref_order) in TEST_2_TO_1
        @testset "$name" begin
            for op in ops
                @testset "$op" begin
                    @test classify_2_to_1(op) == ref_order
                end
            end
        end
    end
end;

## 1-to-2

function classify_1_to_2(f, x; atol)
    d1 = derivative(f, x)
    d2 = second_derivative(f, x)
    ∂f₁∂x = d1[1]
    ∂f₂∂x = d1[2]
    ∂²f₁∂x² = d2[1]
    ∂²f₂∂x² = d2[2]

    first_out = differentiability(∂f₁∂x, ∂²f₁∂x²; atol)
    first_out == error_order && @warn "Weird behavior w.r.t. first output" f x ∂f₁∂x ∂²f₁∂x²
    first_out = differentiability(∂f₂∂x, ∂²f₂∂x²; atol)
    first_out == error_order &&
        @warn "Weird behavior w.r.t. second output" f x ∂f₂∂x ∂²f₂∂x²
    return (first_arg, second_arg)
end

function classify_1_to_2(op::Symbol; atol=1e-5, trials=100)
    f = sym2fn(op)
    try
        return maximum(classify_1_to_1(f, random_input(f); atol) for _ in 1:trials)
    catch e
        @warn "Classification of 1-to-2 operator `$op` failed" e
        return (error_order, error_order)
    end
end

const TEST_1_TO_2 = (
    ("ss", ops_1_to_2_ss, (second_order, second_order)),
    ("sf", ops_1_to_2_sf, (second_order, first_order)),
    ("fs", ops_1_to_2_fs, (first_order, second_order)),
    ("ff", ops_1_to_2_ff, (first_order, first_order)),
    ("sz", ops_1_to_2_sz, (second_order, zero_order)),
    ("zs", ops_1_to_2_zs, (zero_order, second_order)),
    ("fz", ops_1_to_2_fz, (first_order, zero_order)),
    ("zf", ops_1_to_2_zf, (zero_order, second_order)),
    ("zz", ops_1_to_2_zz, (zero_order, zero_order)),
)
@testset verbose = true "1-to-2" begin
    @testset "All operators covered" begin
        all_ops = union([ops for (name, ops, ref_order) in TEST_1_TO_2]...)
        @test Set(all_ops) == Set(ops_1_to_2)
    end
    for (name, ops, ref_order) in TEST_1_TO_2
        @testset "$name" begin
            for op in ops
                @testset "$op" begin
                    @test classify_1_to_2(op) == ref_order
                end
            end
        end
    end
end;
