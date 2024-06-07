# Test construction and conversions of internal tracer types
using SparseConnectivityTracer: ConnectivityTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: inputs, primal, tracer, myempty
using SparseConnectivityTracer:
    SimpleVectorIndexSetPattern, SimpleMatrixIndexSetPattern, CombinedVectorAndMatrixPattern
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

const PATTERNS = (
    (
        SimpleVectorIndexSetPattern{BitSet},
        CombinedVectorAndMatrixPattern{
            SimpleVectorIndexSetPattern{BitSet},
            SimpleMatrixIndexSetPattern{Set{Tuple{Int,Int}}},
        },
    ),
    (
        SimpleVectorIndexSetPattern{Set{Int}},
        CombinedVectorAndMatrixPattern{
            SimpleVectorIndexSetPattern{Set{Int}},
            SimpleMatrixIndexSetPattern{Set{Tuple{Int,Int}}},
        },
    ),
    (
        SimpleVectorIndexSetPattern{DuplicateVector{Int}},
        CombinedVectorAndMatrixPattern{
            SimpleVectorIndexSetPattern{DuplicateVector{Int}},
            SimpleMatrixIndexSetPattern{DuplicateVector{Tuple{Int,Int}}},
        },
    ),
    (
        SimpleVectorIndexSetPattern{SortedVector{Int}},
        CombinedVectorAndMatrixPattern{
            SimpleVectorIndexSetPattern{SortedVector{Int}},
            SimpleMatrixIndexSetPattern{SortedVector{Tuple{Int,Int}}},
        },
    ),
    # TODO: test on RecursiveSet
)

is_pattern_empty(p::SimpleVectorIndexSetPattern) = isempty(p.inds)
function is_pattern_empty(p::CombinedVectorAndMatrixPattern)
    return isempty(p.first_order) && isempty(p.second_order)
end

is_tracer_empty(t::ConnectivityTracer) = isempty(inputs(t)) && t.isempty
is_tracer_empty(t::GradientTracer)     = isempty(SparseConnectivityTracer.gradient(t)) && t.isempty
is_tracer_empty(t::HessianTracer)      = isempty(SparseConnectivityTracer.gradient(t)) && isempty(SparseConnectivityTracer.hessian(t)) && t.isempty
is_tracer_empty(d::Dual)               = is_tracer_empty(tracer(d))

@testset "Pattern types $S1, $S2" for (S1, S2) in PATTERNS
    C = ConnectivityTracer{S1}
    G = GradientTracer{S1}
    H = HessianTracer{S2}

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

        # 1-arg
        BO = similar(B)
        @test eltype(B) == T
        @test size(B) == (2, 3)
        @test all(is_tracer_empty, B)
        if T <: Dual
            @test all(d -> eltype(primal(d)) == TD, B)
        end

        # 2-arg from matrix of tracers
        BD = similar(B, T)
        @test eltype(BD) == T
        @test size(BD) == (2, 3)
        @test all(is_tracer_empty, BD)
        if T <: Dual
            @test all(d -> eltype(primal(d)) == TD, BD)
        end

        # 2-arg from matrix of tracers, custom size
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
    @testset "Constant construction with $f" for f in (
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

    # Test type conversions
    @testset "Type conversion with $f" for f in (big, widen, float)
        @testset "First order $T" for T in (C, G, H)
            @test f(T) == T
        end
        @testset "Dual with $T" for T in (C, G, H)
            @testset "$N" for N in (Int, Float32, Irrational)
                N_OUT = f(N)
                D_IN = Dual{N,T}
                D_OUT = Dual{N_OUT,T}
                @test f(D_IN) == D_OUT
            end
        end
    end

    # Type casting
    f_cast(x::T) where {T} = T(x)
    @testset "Type casting on $T" for T in (C, G, H)
        t_in = myempty(T)
        t_out = f_cast(t_in)
        @test isa(t_out, T)
    end
end
