using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: tracer, trace_input, inputs, empty
using SparseConnectivityTracer: RecursiveSet, SortedVector
using Test

@testset "Set type $S" for S in
                           (BitSet, Set{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64})
    CT = ConnectivityTracer{S}
    JT = JacobianTracer{S}

    x = rand(3)
    xt = trace_input(CT, x)

    # Matrix multiplication
    A = rand(1, 3)
    yt = only(A * xt)
    @test connectivity_pattern(x -> only(A * x), x, S) ≈ [1 1 1]

    # Custom functions
    f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
    yt = f(xt)

    @test connectivity_pattern(f, x, S) ≈ [1 0 0; 1 1 0; 0 0 1]
    @test jacobian_pattern(f, x, S) ≈ [1 0 0; 1 1 0; 0 0 1]

    @test connectivity_pattern(identity, rand(), S) ≈ [1;;]
    @test jacobian_pattern(identity, rand(), S) ≈ [1;;]
    @test connectivity_pattern(Returns(1), 1, S) ≈ [0;;]
    @test jacobian_pattern(Returns(1), 1, S) ≈ [0;;]

    # Test JacobianTracer on functions with zero derivatives
    x = rand(2)
    g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
    @test connectivity_pattern(g, x, S) ≈ [1 1; 1 1; 1 1]
    @test jacobian_pattern(g, x, S) ≈ [1 1; 0 0; 1 0]

    # Code coverage
    @test connectivity_pattern(x -> [sincos(x)...], 1, S) ≈ [1; 1]
    @test connectivity_pattern(typemax, 1, S) ≈ [0;;]
    @test connectivity_pattern(x -> x^(2//3), 1, S) ≈ [1;;]
    @test connectivity_pattern(x -> (2//3)^x, 1, S) ≈ [1;;]
    @test connectivity_pattern(x -> x^ℯ, 1, S) ≈ [1;;]
    @test connectivity_pattern(x -> ℯ^x, 1, S) ≈ [1;;]
    @test connectivity_pattern(x -> round(x, RoundNearestTiesUp), 1, S) ≈ [1;;]

    @test jacobian_pattern(x -> [sincos(x)...], 1, S) ≈ [1; 1]
    @test jacobian_pattern(typemax, 1, S) ≈ [0;;]
    @test jacobian_pattern(x -> x^(2//3), 1, S) ≈ [1;;]
    @test jacobian_pattern(x -> (2//3)^x, 1, S) ≈ [1;;]
    @test jacobian_pattern(x -> x^ℯ, 1, S) ≈ [1;;]
    @test jacobian_pattern(x -> ℯ^x, 1, S) ≈ [1;;]
    @test jacobian_pattern(x -> round(x, RoundNearestTiesUp), 1, S) ≈ [0;;]

    # Base.show
    @test_reference "references/show/ConnectivityTracer_$S.txt" repr(
        "text/plain", tracer(CT, 2)
    )
    @test_reference "references/show/JacobianTracer_$S.txt" repr(
        "text/plain", tracer(JT, 2)
    )
end
