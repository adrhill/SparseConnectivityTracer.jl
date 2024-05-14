using SparseConnectivityTracer
using SparseConnectivityTracer: GlobalHessianTracer, tracer, trace_input, empty
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

@testset "Default hessian_pattern" begin
    h = hessian_pattern(x -> x[1] / x[2] + x[3] / 1 + 1 / x[4], rand(4))
    @test h ≈ [
        0 1 0 0
        1 1 0 0
        0 0 0 0
        0 0 0 1
    ]
end

@testset "Set type $G" for G in (
    BitSet, Set{UInt64}, DuplicateVector{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64}
)
    I = eltype(G)
    H = Set{Tuple{I,I}}
    HT = GlobalHessianTracer{G,H}

    @test hessian_pattern(identity, rand(), G, H) ≈ [0;;]
    @test hessian_pattern(sqrt, rand(), G, H) ≈ [1;;]

    @test hessian_pattern(x -> 1 * x, rand(), G, H) ≈ [0;;]
    @test hessian_pattern(x -> x * 1, rand(), G, H) ≈ [0;;]

    # Code coverage
    @test hessian_pattern(typemax, 1, G, H) ≈ [0;;]
    @test hessian_pattern(x -> x^(2im), 1, G, H) ≈ [1;;]
    @test hessian_pattern(x -> (2im)^x, 1, G, H) ≈ [1;;]
    @test hessian_pattern(x -> x^(2//3), 1, G, H) ≈ [1;;]
    @test hessian_pattern(x -> (2//3)^x, 1, G, H) ≈ [1;;]
    @test hessian_pattern(x -> x^ℯ, 1, G, H) ≈ [1;;]
    @test hessian_pattern(x -> ℯ^x, 1, G, H) ≈ [1;;]
    @test hessian_pattern(x -> round(x, RoundNearestTiesUp), 1, G, H) ≈ [0;;]

    h = hessian_pattern(x -> x[1] / x[2] + x[3] / 1 + 1 / x[4], rand(4), G, H)
    @test h ≈ [
        0 1 0 0
        1 1 0 0
        0 0 0 0
        0 0 0 1
    ]

    h = hessian_pattern(x -> x[1] * x[2] + x[3] * 1 + 1 * x[4], rand(4), G, H)
    @test h ≈ [
        0 1 0 0
        1 0 0 0
        0 0 0 0
        0 0 0 0
    ]

    h = hessian_pattern(x -> (x[1] * x[2]) * (x[3] * x[4]), rand(4), G, H)
    @test h ≈ [
        0 1 1 1
        1 0 1 1
        1 1 0 1
        1 1 1 0
    ]

    h = hessian_pattern(x -> (x[1] + x[2]) * (x[3] + x[4]), rand(4), G, H)
    @test h ≈ [
        0 0 1 1
        0 0 1 1
        1 1 0 0
        1 1 0 0
    ]

    h = hessian_pattern(x -> (x[1] + x[2] + x[3] + x[4])^2, rand(4), G, H)
    @test h ≈ [
        1 1 1 1
        1 1 1 1
        1 1 1 1
        1 1 1 1
    ]

    h = hessian_pattern(x -> 1 / (x[1] + x[2] + x[3] + x[4]), rand(4), G, H)
    @test h ≈ [
        1 1 1 1
        1 1 1 1
        1 1 1 1
        1 1 1 1
    ]

    h = hessian_pattern(x -> (x[1] - x[2]) + (x[3] - 1) + (1 - x[4]), rand(4), G, H)
    @test h ≈ [
        0 0 0 0
        0 0 0 0
        0 0 0 0
        0 0 0 0
    ]

    h = hessian_pattern(x -> copysign(x[1] * x[2], x[3] * x[4]), rand(4), G, H)
    @test h ≈ [
        0 1 0 0
        1 0 0 0
        0 0 0 0
        0 0 0 0
    ]

    h = hessian_pattern(x -> div(x[1] * x[2], x[3] * x[4]), rand(4), G, H)
    @test h ≈ [
        0 0 0 0
        0 0 0 0
        0 0 0 0
        0 0 0 0
    ]

    h = hessian_pattern(x -> sum(sincosd(x)), 1.0, G, H)
    @test h ≈ [1;;]

    h = hessian_pattern(x -> sum(diff(x) .^ 3), rand(4), G, H)
    @test h ≈ [
        1 1 0 0
        1 1 1 0
        0 1 1 1
        0 0 1 1
    ]

    x = rand(5)
    foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
    h = hessian_pattern(foo, x, G, H)
    @test h ≈ [
        0 0 0 0 0
        0 0 1 0 0
        0 1 0 0 0
        0 0 0 1 0
        0 0 0 0 0
    ]

    bar(x) = foo(x) + x[2]^x[5]
    h = hessian_pattern(bar, x, G, H)
    @test h ≈ [
        0 0 0 0 0
        0 1 1 0 1
        0 1 0 0 0
        0 0 0 1 0
        0 1 0 0 1
    ]
end
