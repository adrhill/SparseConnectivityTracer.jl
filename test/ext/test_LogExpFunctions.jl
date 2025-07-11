using SparseConnectivityTracer
using LogExpFunctions
using Test

# Load definitions of GRADIENT_TRACERS and HESSIAN_TRACERS
include("../tracers_definitions.jl")

lef_1_to_1_pos_input = (
    xlogx,
    logistic,
    logit,
    log1psq,
    log1pexp,
    logexpm1,
    softplus,
    invsoftplus,
    log1pmx,
    logmxp1,
    logcosh,
    logabssinh,
    cloglog,
    cexpexp,
    loglogistic,
    log1mlogistic,
)
lef_1_to_1_neg_input = (log1mexp, log2mexp, logitexp, logit1mexp)
lef_1_to_1 = union(lef_1_to_1_pos_input, lef_1_to_1_neg_input)
lef_2_to_1 = (xlogy, xlog1py, xexpy, logaddexp, logsubexp)

@testset "Jacobian Global" begin
    detector = TracerSparsityDetector()
    J(f, x) = jacobian_sparsity(f, x, detector)

    @testset "1-to-1 functions" begin
        @testset "$f" for f in lef_1_to_1
            @test J(x -> f(x[1]), rand(2)) == [1 0]
        end
    end
    @testset "2-to-1 functions" begin
        @testset "$f" for f in lef_2_to_1
            @test J(x -> f(x[1], x[2]), rand(3)) == [1 1 0]
        end
    end
end

@testset "Jacobian Local" begin
    detector = TracerLocalSparsityDetector()
    J(f, x) = jacobian_sparsity(f, x, detector)

    @testset "1-to-1 functions" begin
        @testset "$f" for f in lef_1_to_1_pos_input
            @test J(x -> f(x[1]), [0.5, 1.0]) == [1 0]
        end
        @testset "$f" for f in lef_1_to_1_neg_input
            @test J(x -> f(x[1]), [-0.5, 1.0]) == [1 0]
        end
    end
    @testset "2-to-1 functions" begin
        @testset "$f" for f in lef_2_to_1
            @test J(x -> f(x[1], x[2]), [0.5, 1.0, 2.0]) == [1 1 0]
        end
    end
end

@testset "Hessian Global" begin
    detector = TracerSparsityDetector()
    H(f, x) = hessian_sparsity(f, x, detector)

    @testset "1-to-1 functions" begin
        @testset "$f" for f in lef_1_to_1
            @test H(x -> f(x[1]), rand(2)) == [1 0; 0 0]
        end
    end
    @testset "2-to-1 functions" begin
        @testset "$f" for f in lef_2_to_1
            @test H(x -> f(x[1], x[2]), rand(3)) == [1 1 0; 1 1 0; 0 0 0]
        end
    end
end

@testset "Hessian Local" begin
    detector = TracerLocalSparsityDetector()
    H(f, x) = hessian_sparsity(f, x, detector)

    @testset "1-to-1 functions" begin
        @testset "$f" for f in lef_1_to_1_pos_input
            @test H(x -> f(x[1]), [0.5, 1.0]) == [1 0; 0 0]
        end
        @testset "$f" for f in lef_1_to_1_neg_input
            @test H(x -> f(x[1]), [-0.5, 1.0]) == [1 0; 0 0]
        end
    end
    @testset "2-to-1 functions" begin
        @testset "$f" for f in lef_2_to_1
            @test H(x -> f(x[1], x[2]), [0.5, 1.0, 2.0]) == [1 1 0; 1 1 0; 0 0 0]
        end
    end
end
