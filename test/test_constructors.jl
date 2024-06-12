# Test construction and conversions of internal tracer types
using SparseConnectivityTracer:
    AbstractTracer, ConnectivityTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: inputs, primal, tracer, myempty, name
using SparseConnectivityTracer:
    IndexSetVectorPattern, IndexSetMatrixPattern, CombinedPattern
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

VECTOR_PATTERNS = (
    IndexSetVectorPattern{Int,BitSet},
    IndexSetVectorPattern{Int,Set{Int}},
    IndexSetVectorPattern{Int,DuplicateVector{Int}},
    IndexSetVectorPattern{Int,SortedVector{Int}},
)

VECTOR_AND_MATRIX_PATTERNS = (
    CombinedPattern{
        IndexSetVectorPattern{Int,BitSet},IndexSetMatrixPattern{Int,Set{Tuple{Int,Int}}}
    },
    CombinedPattern{
        IndexSetVectorPattern{Int,Set{Int}},IndexSetMatrixPattern{Int,Set{Tuple{Int,Int}}}
    },
    CombinedPattern{
        IndexSetVectorPattern{Int,DuplicateVector{Int}},
        IndexSetMatrixPattern{Int,DuplicateVector{Tuple{Int,Int}}},
    },
    CombinedPattern{
        IndexSetVectorPattern{Int,SortedVector{Int}},
        IndexSetMatrixPattern{Int,SortedVector{Tuple{Int,Int}}},
    },
    # TODO: test on RecursiveSet
)

is_tracer_empty(t::ConnectivityTracer) = isempty(inputs(t)) && t.isempty
is_tracer_empty(t::GradientTracer)     = isempty(SparseConnectivityTracer.gradient(t)) && t.isempty
is_tracer_empty(t::HessianTracer)      = isempty(SparseConnectivityTracer.gradient(t)) && isempty(SparseConnectivityTracer.hessian(t)) && t.isempty
is_tracer_empty(d::Dual)               = is_tracer_empty(tracer(d))

function test_nested_duals(::Type{T}) where {T<:AbstractTracer}
    # Putting Duals into Duals is prohibited
    t = myempty(T)
    D1 = Dual(1.0, t)
    @test_throws ErrorException D2 = Dual(D1, t)
end

function test_constant_functions(::Type{T}) where {T<:AbstractTracer}
    @testset "$f" for f in (
        zero, one, oneunit, typemin, typemax, eps, floatmin, floatmax, maxintfloat
    )
        t = f(T)
        @test isa(t, T)
        @test is_tracer_empty(t)
    end
end

function test_constant_functions(::Type{D}) where {P,T,D<:Dual{P,T}}
    @testset "$f" for f in (
        zero, one, oneunit, typemin, typemax, eps, floatmin, floatmax, maxintfloat
    )
        d = f(D)
        @test isa(d, D)
        @test is_tracer_empty(d)
        @test primal(d) == f(P)
    end
end

function test_type_conversion_functions(::Type{T}) where {T}
    @testset "$f" for f in (big, widen, float)
        test_type_conversion_functions(T, f)
    end
end
function test_type_conversion_functions(::Type{T}, f::Function) where {T<:AbstractTracer}
    @test f(T) == T
end
function test_type_conversion_functions(::Type{D}, f::Function) where {P,T,D<:Dual{P,T}}
    @testset "Primal type $P_IN" for P_IN in (Int, Float32, Irrational)
        P_OUT = f(P_IN)

        # Note that this tests Dual{P_IN,T}, not Dual{P,T} 
        D_IN = Dual{P_IN,T}
        D_OUT = Dual{P_OUT,T}
        @test f(D_IN) == D_OUT
    end
end

function test_type_casting(::Type{T}) where {T<:AbstractTracer}
    t_in = myempty(T)
    @testset "$(name(T)) to $(name(T))" begin
        t_out = T(t_in)
        @test t_out isa T
        @test is_tracer_empty(t_out)
    end
    @testset "$N to $(name(T))" for N in (Int, Float32, Irrational)
        t_out = T(one(N))
        @test t_out isa T
        @test is_tracer_empty(t_out)
    end
end

function test_type_casting(::Type{D}) where {P,T,D<:Dual{P,T}}
    d_in = Dual(one(P), myempty(T))
    @testset "$(name(D)) to $(name(D))" begin
        d_out = D(d_in)
        @test primal(d_out) == primal(d_in)
        @test tracer(d_out) isa T
        @test is_tracer_empty(d_out)
    end
    @testset "$P2 to $(name(D))" for P2 in (Int, Float32, Irrational)
        p_in = one(P2)
        d_out = D(p_in)
        @test primal(d_out) == P(p_in)
        @test tracer(d_out) isa T
        @test is_tracer_empty(d_out)
    end
end

function test_similar(::Type{T}) where {T<:AbstractTracer}
    A = rand(Int, 2, 3)

    # 2-arg from matrix of Reals
    B = similar(A, T)
    @test eltype(B) == T
    @test size(B) == (2, 3)
    @test all(is_tracer_empty, B)

    # 1-arg from matrix of tracers
    B1 = similar(B)
    @test eltype(B1) == T
    @test size(B1) == (2, 3)
    @test all(is_tracer_empty, B1)

    # 2-arg from matrix of tracers
    B2 = similar(B, T)
    @test eltype(B2) == T
    @test size(B2) == (2, 3)
    @test all(is_tracer_empty, B2)

    # 2-arg from matrix of tracers, custom size
    B3 = similar(B, 4, 5)
    @test eltype(B3) == T
    @test size(B3) == (4, 5)
    @test all(is_tracer_empty, B3)

    # 3-arg from matrix of Reals
    B4 = similar(A, T, 4, 5)
    @test eltype(B4) == T
    @test size(B4) == (4, 5)
    @test all(is_tracer_empty, B4)

    # 3-arg from matrix of tracers
    B5 = similar(B, T, 5, 6)
    @test eltype(B5) == T
    @test size(B5) == (5, 6)
    @test all(is_tracer_empty, B5)
end

function test_similar(::Type{D}) where {P,T,D<:Dual{P,T}}
    # Test `similar`
    P2 = Float16   # using something different than P
    @test P2 != P # this is important for following tests

    A = rand(P2, 2, 3)

    # 2-arg from matrix of Reals P2
    B = similar(A, D)
    @test eltype(B) == D
    @test size(B) == (2, 3)
    @test all(is_tracer_empty, B)
    @test all(d -> primal(d) isa P, B)

    # 1-arg from matrix of tracers
    B1 = similar(B)
    @test eltype(B1) == D
    @test size(B1) == (2, 3)
    @test all(is_tracer_empty, B1)
    @test all(d -> primal(d) isa P, B1)

    # 2-arg from matrix of tracers
    B2 = similar(B, D)
    @test eltype(B2) == D
    @test size(B2) == (2, 3)
    @test all(is_tracer_empty, B2)
    @test all(d -> primal(d) isa P, B2)

    # 2-arg from matrix of tracers, custom size
    B3 = similar(B, 4, 5)
    @test eltype(B3) == D
    @test size(B3) == (4, 5)
    @test all(is_tracer_empty, B3)
    @test all(d -> primal(d) isa P, B3)

    # 3-arg from matrix of Reals
    B4 = similar(A, D, 4, 5)
    @test eltype(B4) == D
    @test size(B4) == (4, 5)
    @test all(is_tracer_empty, B4)
    @test all(d -> primal(d) isa P, B4)

    # 3-arg from matrix of tracers
    B5 = similar(B, D, 5, 6)
    @test eltype(B5) == D
    @test size(B5) == (5, 6)
    @test all(is_tracer_empty, B5)
    @test all(d -> primal(d) isa P, B5)
end

@testset "First order tracers" begin
    @testset "Pattern $P" for P in VECTOR_PATTERNS
        C = ConnectivityTracer{P}
        G = GradientTracer{P}

        N = Float32
        DC = Dual{N,C}
        DG = Dual{N,G}

        TS = (C, G, DC, DG)

        @testset "Nested Duals on $(name(T))" for T in (C, G)
            test_nested_duals(T)
        end

        @testset "Constant functions on $(name(T))" for T in TS
            test_constant_functions(T)
        end
        @testset "Type conversions on $(name(T))" for T in TS
            test_type_conversion_functions(T)
        end
        @testset "Type casting on $(name(T))" for T in TS
            test_type_casting(T)
        end
        @testset "similar on $(name(T))" for T in TS
            test_similar(T)
        end
    end
end

@testset "Second order tracers" begin
    @testset "Pattern $P" for P in VECTOR_AND_MATRIX_PATTERNS
        H = HessianTracer{P}

        N = Float32
        DH = Dual{N,H}

        TS = (H, DH)

        @testset "Nested Duals on HessianTracer" test_nested_duals(H)

        @testset "Constant functions on $(name(T))" for T in TS
            test_constant_functions(T)
        end
        @testset "Type conversions on $(name(T))" for T in TS
            test_type_conversion_functions(T)
        end
        @testset "Type casting on $(name(T))" for T in TS
            test_type_casting(T)
        end
        @testset "similar on $(name(T))" for T in TS
            test_similar(T)
        end
    end
end
