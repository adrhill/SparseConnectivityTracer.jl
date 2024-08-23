using SparseConnectivityTracer
using DataInterpolations
using DataInterpolations: AbstractInterpolation
using Test

#===========#
# Test data #
#===========#

t = [0.0, 1.0, 2.5, 4.0];
n = length(t)

uv = rand(n); # vector
um = rand(2, n); # matrix

tquery = 2.0

#==================#
# Test definitions #
#==================#

struct InterpolationTest{N,I<:AbstractInterpolation} # N = output dim. of interpolation
    interp::I
    is_der1_zero::Bool
    is_der2_zero::Bool
end
function InterpolationTest(
    N, interp::I; is_der1_zero=false, is_der2_zero=false
) where {I<:AbstractInterpolation}
    return InterpolationTest{N,I}(interp, is_der1_zero, is_der2_zero)
end
name(t::InterpolationTest{N}) where {N} = "$N-dim $(typeof(t.interp))"

# scalar interpolations
function test_interpolation(t::InterpolationTest{1})
    Jref = [Int(!t.is_der1_zero);;]
    Href = [Int(!t.is_der2_zero);;]

    @testset "Jacobian Global" begin
        J = jacobian_sparsity(t.interp, tquery, TracerSparsityDetector())
        @test J ≈ Jref
    end
    @testset "Jacobian Local" begin
        J = jacobian_sparsity(t.interp, tquery, TracerLocalSparsityDetector())
        @test J ≈ Jref
    end
    @testset "Hessian Global" begin
        H = hessian_sparsity(t.interp, tquery, TracerSparsityDetector())
        @test H ≈ Href
    end
    @testset "Hessian Local" begin
        H = hessian_sparsity(t.interp, tquery, TracerLocalSparsityDetector())
        @test H ≈ Href
    end
end

# vector-valued interpolations
function test_interpolation(t::InterpolationTest{N}) where {N} # N ≠ 1
    Jref = t.is_der1_zero ? zeros(N) : ones(N)
    Href = t.is_der2_zero ? zeros(N) : ones(N)

    @testset "Jacobian Global" begin
        J = jacobian_sparsity(t.interp, tquery, TracerSparsityDetector())
        @test J ≈ Jref
    end
    @testset "Jacobian Local" begin
        J = jacobian_sparsity(t.interp, tquery, TracerLocalSparsityDetector())
        @test J ≈ Jref
    end
    @testset "Hessian Global" begin
        H = hessian_sparsity(t.interp, tquery, TracerSparsityDetector())
        @test H ≈ Href
    end
    @testset "Hessian Local" begin
        H = hessian_sparsity(t.interp, tquery, TracerLocalSparsityDetector())
        @test H ≈ Href
    end
end

#===========#
# Run tests #
#===========#

interpolation_tests = (
    InterpolationTest(
        1, ConstantInterpolation(uv, t); is_der1_zero=true, is_der2_zero=true
    ),
    InterpolationTest(1, LinearInterpolation(uv, t); is_der2_zero=true),
    InterpolationTest(1, QuadraticInterpolation(uv, t)),
    InterpolationTest(1, LagrangeInterpolation(uv, t)),
    InterpolationTest(1, AkimaInterpolation(uv, t)),
    InterpolationTest(1, QuadraticSpline(uv, t)),
    InterpolationTest(1, CubicSpline(uv, t)),
)


@testset "Test interpolations" begin
    @testset "$(name(t))" for t in interpolation_tests
        test_interpolation(t)
    end
end
