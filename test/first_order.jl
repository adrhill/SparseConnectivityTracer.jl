using SparseConnectivityTracer
using SparseConnectivityTracer:
    ConnectivityTracer, GlobalGradientTracer, tracer, trace_input, empty
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

@testset "Set type $G" for G in (
    BitSet, Set{UInt64}, DuplicateVector{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64}
)
    CT = ConnectivityTracer{G}
    JT = GlobalGradientTracer{G}

    x = rand(3)
    xt = trace_input(CT, x)

    # Matrix multiplication
    A = rand(1, 3)
    yt = only(A * xt)
    @test connectivity_pattern(x -> only(A * x), x, G) ≈ [1 1 1]

    # Custom functions
    f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
    yt = f(xt)

    @test connectivity_pattern(f, x, G) ≈ [1 0 0; 1 1 0; 0 0 1]
    @test jacobian_pattern(f, x, G) ≈ [1 0 0; 1 1 0; 0 0 1]

    @test connectivity_pattern(identity, rand(), G) ≈ [1;;]
    @test jacobian_pattern(identity, rand(), G) ≈ [1;;]
    @test connectivity_pattern(Returns(1), 1, G) ≈ [0;;]
    @test jacobian_pattern(Returns(1), 1, G) ≈ [0;;]

    # Test GlobalGradientTracer on functions with zero derivatives
    x = rand(2)
    g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
    @test connectivity_pattern(g, x, G) ≈ [1 1; 1 1; 1 1]
    @test jacobian_pattern(g, x, G) ≈ [1 1; 0 0; 1 0]

    # Code coverage
    @test connectivity_pattern(x -> [sincos(x)...], 1, G) ≈ [1; 1]
    @test connectivity_pattern(typemax, 1, G) ≈ [0;;]
    @test connectivity_pattern(x -> x^(2//3), 1, G) ≈ [1;;]
    @test connectivity_pattern(x -> (2//3)^x, 1, G) ≈ [1;;]
    @test connectivity_pattern(x -> x^ℯ, 1, G) ≈ [1;;]
    @test connectivity_pattern(x -> ℯ^x, 1, G) ≈ [1;;]
    @test connectivity_pattern(x -> round(x, RoundNearestTiesUp), 1, G) ≈ [1;;]

    @test jacobian_pattern(x -> [sincos(x)...], 1, G) ≈ [1; 1]
    @test jacobian_pattern(typemax, 1, G) ≈ [0;;]
    @test jacobian_pattern(x -> x^(2//3), 1, G) ≈ [1;;]
    @test jacobian_pattern(x -> (2//3)^x, 1, G) ≈ [1;;]
    @test jacobian_pattern(x -> x^ℯ, 1, G) ≈ [1;;]
    @test jacobian_pattern(x -> ℯ^x, 1, G) ≈ [1;;]
    @test jacobian_pattern(x -> round(x, RoundNearestTiesUp), 1, G) ≈ [0;;]
end
