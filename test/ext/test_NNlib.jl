using SparseConnectivityTracer
using NNlib: NNlib
using Test

NNLIB_ACTIVATIONS_S = (
    NNlib.σ,
    NNlib.celu,
    NNlib.elu,
    NNlib.gelu,
    NNlib.hardswish,
    NNlib.lisht,
    NNlib.logσ,
    NNlib.logcosh,
    NNlib.mish,
    NNlib.selu,
    NNlib.softplus,
    NNlib.softsign,
    NNlib.swish,
    NNlib.sigmoid_fast,
    NNlib.tanhshrink,
    NNlib.tanh_fast,
)
NNLIB_ACTIVATIONS_F = (
    NNlib.hardσ,
    NNlib.hardtanh,
    NNlib.leakyrelu,
    NNlib.relu,
    NNlib.relu6,
    NNlib.softshrink,
    NNlib.trelu,
)
NNLIB_ACTIVATIONS = union(NNLIB_ACTIVATIONS_S, NNLIB_ACTIVATIONS_F)

@testset "Jacobian Global" begin
    method = TracerSparsityDetector()
    J(f, x) = jacobian_sparsity(f, x, method)

    @testset "$f" for f in NNLIB_ACTIVATIONS
        @test J(f, 1) ≈ [1;;]
    end
end

@testset "Jacobian Local" begin
    method = TracerLocalSparsityDetector()
    J(f, x) = jacobian_sparsity(f, x, method)

    @test J(NNlib.relu, -1) ≈ [0;;]
    @test J(NNlib.relu, 1) ≈ [1;;]
    @test J(NNlib.elu, -1) ≈ [1;;]
    @test J(NNlib.elu, 1) ≈ [1;;]
    @test J(NNlib.celu, -1) ≈ [1;;]
    @test J(NNlib.celu, 1) ≈ [1;;]
    @test J(NNlib.selu, -1) ≈ [1;;]
    @test J(NNlib.selu, 1) ≈ [1;;]

    @test J(NNlib.relu6, -1) ≈ [0;;]
    @test J(NNlib.relu6, 1) ≈ [1;;]
    @test J(NNlib.relu6, 7) ≈ [0;;]

    @test J(NNlib.trelu, 0.9) ≈ [0;;]
    @test J(NNlib.trelu, 1.1) ≈ [1;;]

    @test J(NNlib.swish, -5) ≈ [1;;]
    @test J(NNlib.swish, 0) ≈ [1;;]
    @test J(NNlib.swish, 5) ≈ [1;;]

    @test J(NNlib.hardswish, -5) ≈ [0;;]
    @test J(NNlib.hardswish, 0) ≈ [1;;]
    @test J(NNlib.hardswish, 5) ≈ [1;;]

    @test J(NNlib.hardσ, -4) ≈ [0;;]
    @test J(NNlib.hardσ, 0) ≈ [1;;]
    @test J(NNlib.hardσ, 4) ≈ [0;;]

    @test J(NNlib.hardtanh, -2) ≈ [0;;]
    @test J(NNlib.hardtanh, 0) ≈ [1;;]
    @test J(NNlib.hardtanh, 2) ≈ [0;;]

    @test J(NNlib.softshrink, -1) ≈ [1;;]
    @test J(NNlib.softshrink, 0) ≈ [0;;]
    @test J(NNlib.softshrink, 1) ≈ [1;;]
end

@testset "Global Hessian" begin
    method = TracerSparsityDetector()
    H(f, x) = hessian_sparsity(f, x, method)

    @testset "First-order differentiable" begin
        @testset "$f" for f in NNLIB_ACTIVATIONS_F
            @test H(f, 1) ≈ [0;;]
        end
    end
    @testset "Second-order differentiable" begin
        @testset "$f" for f in NNLIB_ACTIVATIONS_S
            @test H(f, 1) ≈ [1;;]
        end
    end
end

@testset "Local Hessian" begin
    method = TracerLocalSparsityDetector()
    H(f, x) = hessian_sparsity(f, x, method)

    @test H(NNlib.relu, -1) ≈ [0;;]
    @test H(NNlib.relu, 1) ≈ [0;;]
    @test H(NNlib.elu, -1) ≈ [1;;]
    @test H(NNlib.elu, 1) ≈ [0;;]
    @test H(NNlib.celu, -1) ≈ [1;;]
    @test H(NNlib.celu, 1) ≈ [0;;]
    @test H(NNlib.selu, -1) ≈ [1;;]
    @test H(NNlib.selu, 1) ≈ [0;;]

    @test H(NNlib.relu6, -1) ≈ [0;;]
    @test H(NNlib.relu6, 1) ≈ [0;;]
    @test H(NNlib.relu6, 7) ≈ [0;;]

    @test H(NNlib.trelu, 0.9) ≈ [0;;]
    @test H(NNlib.trelu, 1.1) ≈ [0;;]

    @test H(NNlib.swish, -5) ≈ [1;;]
    @test H(NNlib.swish, 0) ≈ [1;;]
    @test H(NNlib.swish, 5) ≈ [1;;]

    @test H(NNlib.hardswish, -5) ≈ [0;;]
    @test H(NNlib.hardswish, 0) ≈ [1;;]
    @test H(NNlib.hardswish, 5) ≈ [0;;]

    @test H(NNlib.hardσ, -4) ≈ [0;;]
    @test H(NNlib.hardσ, 0) ≈ [0;;]
    @test H(NNlib.hardσ, 4) ≈ [0;;]

    @test H(NNlib.hardtanh, -2) ≈ [0;;]
    @test H(NNlib.hardtanh, 0) ≈ [0;;]
    @test H(NNlib.hardtanh, 2) ≈ [0;;]

    @test H(NNlib.softshrink, -1) ≈ [0;;]
    @test H(NNlib.softshrink, 0) ≈ [0;;]
    @test H(NNlib.softshrink, 1) ≈ [0;;]
end
