using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: tracer, trace_input, inputs, empty
using SparseConnectivityTracer: SortedVector
using Test

@testset "Set type $S" for S in (BitSet, Set{UInt64}, SortedVector{UInt64})
    HT = HessianTracer{S}

    @test hessian_pattern(identity, rand(), S) ≈ [0;;]
    @test hessian_pattern(sqrt, rand(), S) ≈ [1;;]

    @test hessian_pattern(x -> 1 * x, rand(), S) ≈ [0;;]
    @test hessian_pattern(x -> x * 1, rand(), S) ≈ [0;;]

    # Code coverage
    @test hessian_pattern(typemax, 1) ≈ [0;;]
    @test hessian_pattern(x -> x^(2im), 1) ≈ [1;;]
    @test hessian_pattern(x -> (2im)^x, 1) ≈ [1;;]
    @test hessian_pattern(x -> x^(2//3), 1) ≈ [1;;]
    @test hessian_pattern(x -> (2//3)^x, 1) ≈ [1;;]
    @test hessian_pattern(x -> x^ℯ, 1) ≈ [1;;]
    @test hessian_pattern(x -> ℯ^x, 1) ≈ [1;;]
    @test hessian_pattern(x -> round(x, RoundNearestTiesUp), 1) ≈ [0;;]

    H = hessian_pattern(x -> x[1] / x[2] + x[3] / 1 + 1 / x[4], rand(4), S)
    @test H ≈ [
        0 1 0 0
        1 1 0 0
        0 0 0 0
        0 0 0 1
    ]

    H = hessian_pattern(x -> x[1] * x[2] + x[3] * 1 + 1 * x[4], rand(4), S)
    @test H ≈ [
        0 1 0 0
        1 0 0 0
        0 0 0 0
        0 0 0 0
    ]

    H = hessian_pattern(x -> (x[1] * x[2]) * (x[3] * x[4]), rand(4), S)
    @test H ≈ [
        0 1 1 1
        1 0 1 1
        1 1 0 1
        1 1 1 0
    ]

    H = hessian_pattern(x -> (x[1] + x[2]) * (x[3] + x[4]), rand(4), S)
    @test H ≈ [
        0 0 1 1
        0 0 1 1
        1 1 0 0
        1 1 0 0
    ]

    H = hessian_pattern(x -> (x[1] + x[2] + x[3] + x[4])^2, rand(4), S)
    @test H ≈ [
        1 1 1 1
        1 1 1 1
        1 1 1 1
        1 1 1 1
    ]

    H = hessian_pattern(x -> 1 / (x[1] + x[2] + x[3] + x[4]), rand(4), S)
    @test H ≈ [
        1 1 1 1
        1 1 1 1
        1 1 1 1
        1 1 1 1
    ]

    H = hessian_pattern(x -> (x[1] - x[2]) + (x[3] - 1) + (1 - x[4]), rand(4), S)
    @test H ≈ [
        0 0 0 0
        0 0 0 0
        0 0 0 0
        0 0 0 0
    ]

    H = hessian_pattern(x -> copysign(x[1] * x[2], x[3] * x[4]), rand(4), S)
    @test H ≈ [
        0 1 0 0
        1 0 0 0
        0 0 0 0
        0 0 0 0
    ]

    H = hessian_pattern(x -> div(x[1] * x[2], x[3] * x[4]), rand(4), S)
    @test H ≈ [
        0 0 0 0
        0 0 0 0
        0 0 0 0
        0 0 0 0
    ]

    H = hessian_pattern(x -> sum(sincosd(x)), 1.0, S)
    @test H ≈ [1;;]

    H = hessian_pattern(x -> sum(diff(x) .^ 3), rand(4), S)
    @test H ≈ [
        1 1 0 0
        1 1 1 0
        0 1 1 1
        0 0 1 1
    ]

    x = rand(5)
    foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
    H = hessian_pattern(foo, x, S)
    @test H ≈ [
        0 0 0 0 0
        0 0 1 0 0
        0 1 0 0 0
        0 0 0 1 0
        0 0 0 0 0
    ]

    bar(x) = foo(x) + x[2]^x[5]
    H = hessian_pattern(bar, x, S)
    @test H ≈ [
        0 0 0 0 0
        0 1 1 0 1
        0 1 0 0 0
        0 0 0 1 0
        0 1 0 0 1
    ]

    # Base.show
    @test_reference "references/show/HessianTracer_$S.txt" repr("text/plain", tracer(HT, 2))
end
