# Test construction and conversions of internal tracer types
using SparseConnectivityTracer: ConnectivityTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: inputs, gradient, hessian, primal, tracer, myempty
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

const FIRST_ORDER_SET_TYPES = (
    BitSet, Set{UInt64}, DuplicateVector{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64}
)

is_tracer_empty(t::ConnectivityTracer) = isempty(inputs(t))
is_tracer_empty(t::GradientTracer)     = isempty(gradient(t))
is_tracer_empty(t::HessianTracer)      = isempty(gradient(t)) && isempty(hessian(t))
is_tracer_empty(d::Dual)               = is_tracer_empty(tracer(d))

@testset "Set type $S1" for S1 in FIRST_ORDER_SET_TYPES
    I = eltype(S1)
    S2 = Set{Tuple{I,I}}

    C = ConnectivityTracer{S1}
    G = GradientTracer{S1}
    H = HessianTracer{S1,S2}

    TD = Float32
    DC = Dual{TD,C}
    DG = Dual{TD,G}
    DH = Dual{TD,H}

    # Putting Duals into Duals is prohibited
    @testset "Nested Duals $T" for T in (C, G, H)
        t = myempty(T)
        D1 = Dual(1.0, t)
        @test_throws ErrorException D2 = Dual(D1, t)
    end

    # Test `similar`
    TA = Float16   # using something different than TD
    @test TA != TD # this is important for following tests

    A = rand(TA, 2, 3)
    @testset "Similar $T" for T in (C, G, H, DC, DG, DH)
        # From matrix of Reals
        B = similar(A, T)
        @test eltype(B) == T
        @test size(B) == (2, 3)
        @test all(is_tracer_empty, B)
        if T <: Dual
            @test all(d -> eltype(primal(d)) == TD, B)
        end

        # From matrix of tracers
        BD = similar(B, T)
        @test eltype(BD) == T
        @test size(BD) == (2, 3)
        @test all(is_tracer_empty, BD)
        if T <: Dual
            @test all(d -> eltype(primal(d)) == TD, BD)
        end

        # From matrix of tracers, custom size
        BD2 = similar(B, 4, 5)
        @test eltype(BD2) == T
        @test size(BD2) == (4, 5)
        @test all(is_tracer_empty, BD2)
        if T <: Dual
            @test all(d -> eltype(primal(d)) == TD, BD2)
        end

        # 3-arg from matrix of Reals
        B = similar(A, T, 4, 5)
        @test eltype(B) == T
        @test size(B) == (4, 5)
        @test all(is_tracer_empty, B)
        if T <: Dual
            @test all(d -> eltype(primal(d)) == TD, B)
        end

        # 3-arg from matrix of tracers
        BD = similar(B, T, 5, 6)
        @test eltype(BD) == T
        @test size(BD) == (5, 6)
        @test all(is_tracer_empty, BD)
        if T <: Dual
            @test all(d -> eltype(primal(d)) == TD, BD)
        end
    end

    # Constant constructors by type
    @testset "$f" for f in (
        zero, one, oneunit, typemin, typemax, eps, floatmin, floatmax, maxintfloat
    )
        @testset "$T" for T in (C, G, H, DC, DG, DH)
            t = f(T)
            @test isa(t, T)
            @test is_tracer_empty(t)
            if T <: Dual
                @test primal(t) == f(TD)
            end
        end
    end
end
