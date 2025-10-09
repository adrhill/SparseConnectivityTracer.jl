# Test construction and conversions of internal tracer types
using SparseConnectivityTracer: AbstractTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: primal, tracer, isemptytracer, myempty
using Test

# Load definitions of GRADIENT_TRACERS and HESSIAN_TRACERS
include("tracers_definitions.jl")

# Pretty-printing of Dual tracers
tracer_name(::Type{T}) where {T <: GradientTracer} = "GradientTracer"
tracer_name(::Type{T}) where {T <: HessianTracer} = "HessianTracer"
tracer_name(::Type{D}) where {P, T, D <: Dual{P, T}} = "Dual-$(tracer_name(T))"

function test_nested_duals(::Type{T}) where {T <: AbstractTracer}
    # Putting Duals into Duals is prohibited
    t = myempty(T)
    D1 = Dual(1.0, t)
    return @test_throws ErrorException D2 = Dual(D1, t)
end

function test_constant_functions(::Type{T}) where {T <: AbstractTracer}
    return @testset "$f" for f in (
            zero, one, oneunit, typemin, typemax, eps, floatmin, floatmax, maxintfloat,
        )
        t = f(T)
        @test isa(t, T)
        @test isemptytracer(t)
    end
end

function test_constant_functions(::Type{D}) where {P, T, D <: Dual{P, T}}
    return @testset "$f" for f in (
            zero, one, oneunit, typemin, typemax, eps, floatmin, floatmax, maxintfloat,
        )
        out = f(D)
        @test out isa D
    end
end

function test_type_conversion_functions(::Type{T}) where {T}
    return @testset "$f" for f in (big, widen, float)
        test_type_conversion_functions(T, f)
    end
end
function test_type_conversion_functions(::Type{T}, f::Function) where {T <: AbstractTracer}
    return @test f(T) == T
end
function test_type_conversion_functions(::Type{D}, f::Function) where {P, T, D <: Dual{P, T}}
    return @testset "Primal type $P_IN" for P_IN in (Int, Float32, Irrational)
        P_OUT = f(P_IN)
        @test f(Dual{P_IN, T}) == P_OUT  # NOTE: this tests Dual{P_IN,T}, not Dual{P,T}
    end
end

function test_type_casting(::Type{T}) where {T <: AbstractTracer}
    t_in = myempty(T)
    @testset "$T to $T" begin
        t_out = T(t_in)
        @test t_out isa T
        @test isemptytracer(t_out)
    end
    return @testset "$N to $T" for N in (Int, Float32, Irrational)
        t_out = T(one(N))
        @test t_out isa T
        @test isemptytracer(t_out)
    end
end

function test_type_casting(::Type{D}) where {P, T, D <: Dual{P, T}}
    d_in = Dual(one(P), myempty(T))
    @testset "$(tracer_name(D)) to $(tracer_name(D))" begin
        d_out = D(d_in)
        @test primal(d_out) == primal(d_in)
        @test tracer(d_out) isa T
        @test isemptytracer(d_out)
    end
    return @testset "$P2 to $(tracer_name(D))" for P2 in (Int, Float32, Irrational)
        p_in = one(P2)
        d_out = D(p_in)
        @test primal(d_out) == P(p_in)
        @test tracer(d_out) isa T
        @test isemptytracer(d_out)
    end
end

function test_similar(::Type{T}) where {T <: AbstractTracer}
    A = rand(Int, 2, 3)

    # 2-arg from matrix of Reals
    B = similar(A, T)
    @test eltype(B) == T
    @test size(B) == (2, 3)

    # 1-arg from matrix of tracers
    B1 = similar(B)
    @test eltype(B1) == T
    @test size(B1) == (2, 3)

    # 2-arg from matrix of tracers
    B2 = similar(B, T)
    @test eltype(B2) == T
    @test size(B2) == (2, 3)

    # 2-arg from matrix of tracers, custom size
    B3 = similar(B, 4, 5)
    @test eltype(B3) == T
    @test size(B3) == (4, 5)

    # 3-arg from matrix of Reals
    B4 = similar(A, T, 4, 5)
    @test eltype(B4) == T
    @test size(B4) == (4, 5)

    # 3-arg from matrix of tracers
    B5 = similar(B, T, 5, 6)
    @test eltype(B5) == T
    return @test size(B5) == (5, 6)
end

function test_similar(::Type{D}) where {P, T, D <: Dual{P, T}}
    # Test `similar`
    P2 = Float16   # using something different than P
    @test P2 != P # this is important for following tests

    A = rand(P2, 2, 3)

    # 2-arg from matrix of Reals P2
    B = similar(A, D)
    @test eltype(B) == D
    @test size(B) == (2, 3)

    # 1-arg from matrix of tracers
    B1 = similar(B)
    @test eltype(B1) == D
    @test size(B1) == (2, 3)

    # 2-arg from matrix of tracers
    B2 = similar(B, D)
    @test eltype(B2) == D
    @test size(B2) == (2, 3)

    # 2-arg from matrix of tracers, custom size
    B3 = similar(B, 4, 5)
    @test eltype(B3) == D
    @test size(B3) == (4, 5)

    # 3-arg from matrix of Reals
    B4 = similar(A, D, 4, 5)
    @test eltype(B4) == D
    @test size(B4) == (4, 5)

    # 3-arg from matrix of tracers
    B5 = similar(B, D, 5, 6)
    @test eltype(B5) == D
    return @test size(B5) == (5, 6)
end

@testset "GradientTracer" begin
    P = Float32
    DUAL_GRADIENT_TRACERS = [Dual{P, T} for T in GRADIENT_TRACERS]
    ALL_GRADIENT_TRACERS = (GRADIENT_TRACERS..., DUAL_GRADIENT_TRACERS...)

    @testset "Nested Duals on HessianTracer" for T in GRADIENT_TRACERS
        test_nested_duals(T)
    end
    @testset "Constant functions on $T" for T in ALL_GRADIENT_TRACERS
        test_constant_functions(T)
    end
    @testset "Type conversions on $T" for T in ALL_GRADIENT_TRACERS
        test_type_conversion_functions(T)
    end
    @testset "Type casting on $T" for T in ALL_GRADIENT_TRACERS
        test_type_casting(T)
    end
    @testset "similar on $T" for T in ALL_GRADIENT_TRACERS
        test_similar(T)
    end
end

@testset "HessianTracer" begin
    P = Float32
    DUAL_HESSIAN_TRACERS = [Dual{P, T} for T in HESSIAN_TRACERS]
    ALL_HESSIAN_TRACERS = (HESSIAN_TRACERS..., DUAL_HESSIAN_TRACERS...)

    @testset "Nested Duals on HessianTracer" for T in HESSIAN_TRACERS
        test_nested_duals(T)
    end
    @testset "Constant functions on $T" for T in ALL_HESSIAN_TRACERS
        test_constant_functions(T)
    end
    @testset "Type conversions on $T" for T in ALL_HESSIAN_TRACERS
        test_type_conversion_functions(T)
    end
    @testset "Type casting on $T" for T in ALL_HESSIAN_TRACERS
        test_type_casting(T)
    end
    @testset "similar on $T" for T in ALL_HESSIAN_TRACERS
        test_similar(T)
    end
end

@testset "Explicit type conversions on Dual" begin
    T = DEFAULT_GRADIENT_TRACER
    t_full = T(BitSet(2))
    t_empty = myempty(T)
    d_full = Dual(1.0, t_full)
    d_empty = Dual(1.0, t_empty)

    @testset "Non-empty tracer" begin
        @testset "$TOUT" for TOUT in (Int, Integer, Float64, Float32)
            @test_throws InexactError TOUT(d_full)
        end
    end
    @testset "Empty tracer" begin
        @testset "$TOUT" for TOUT in (Int, Integer, Float64, Float32)
            out = TOUT(d_empty)
            @test out isa TOUT # not a Dual!
            @test isone(out)
        end
    end
    @testset "Conversions of Reals to Dual" begin
        @testset "$P_out" for P_out in (Float16, Float64, Float32)
            @testset "$P_in" for P_in in (Int, Float64, Float32)
                x = oneunit(P_in)
                D = Dual{P_out, T}
                d = convert(D, x)
                @test primal(d) isa P_out
            end
        end
    end
end
