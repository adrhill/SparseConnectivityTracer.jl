using ADTypes: jacobian_sparsity, hessian_sparsity
using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, HessianTracer, Shared, NotShared
using SparseConnectivityTracer: DEFAULT_GRADIENT_TRACER, DEFAULT_HESSIAN_TRACER, DEFAULT_SHARED_TYPE
using SparseArrays
using Test

@testset "Global" begin
    sd = TracerSparsityDetector()

    x = rand(10)
    y = zeros(9)
    J1 = jacobian_sparsity(diff, x, sd)
    J2 = jacobian_sparsity((y, x) -> y .= diff(x), y, x, sd)
    @test J1 == J2
    @test J1 isa SparseMatrixCSC{Bool, Int}
    @test J2 isa SparseMatrixCSC{Bool, Int}
    @test nnz(J1) == nnz(J2) == 18

    H1 = hessian_sparsity(x -> sum(diff(x)), x, sd)
    @test H1 ≈ zeros(10, 10)

    x = rand(5)
    f(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
    H2 = hessian_sparsity(f, x, sd)
    @test H2 isa SparseMatrixCSC{Bool, Int}
    @test H2 ≈ [
        0 0 0 0 0
        0 0 1 0 0
        0 1 0 0 0
        0 0 0 1 0
        0 0 0 0 0
    ]
end

@testset "Local" begin
    lsd = TracerLocalSparsityDetector()
    fl1(x) = x[1] + x[2] * x[3] + 1 / x[4] + x[2] * max(x[1], x[5])
    HL1 = hessian_sparsity(fl1, [1.0 3.0 5.0 1.0 2.0], lsd)
    @test HL1 ≈ [
        0  0  0  0  0
        0  0  1  0  1
        0  1  0  0  0
        0  0  0  1  0
        0  1  0  0  0
    ]
    HL2 = hessian_sparsity(fl1, [4.0 3.0 5.0 1.0 2.0], lsd)
    @test HL2 ≈ [
        0  1  0  0  0
        1  0  1  0  0
        0  1  0  0  0
        0  0  0  1  0
        0  0  0  0  0
    ]
end

@testset "Constructor" begin
    TG = DEFAULT_GRADIENT_TRACER
    TH = DEFAULT_HESSIAN_TRACER

    @test TracerSparsityDetector() isa TracerSparsityDetector{TG, TH}
    @test TracerLocalSparsityDetector() isa TracerLocalSparsityDetector{TG, TH}

    ## Gradient
    G = Set{Int16}
    @test TracerSparsityDetector(; gradient_pattern_type = G) isa TracerSparsityDetector{GradientTracer{Int16, Set{Int16}}, TH}
    @test TracerLocalSparsityDetector(; gradient_pattern_type = G) isa TracerLocalSparsityDetector{GradientTracer{Int16, Set{Int16}}, TH}
    @test_throws TypeError TracerSparsityDetector(; gradient_pattern_type = Set{Float32})
    @test_throws TypeError TracerLocalSparsityDetector(; gradient_pattern_type = Set{Float32})
    @test_throws MethodError TracerSparsityDetector(; gradient_pattern_type = Dict{Int, Set{Int}})
    @test_throws MethodError TracerLocalSparsityDetector(; gradient_pattern_type = Dict{Int, Set{Int}})

    ## Hessian
    # Dict
    H = Dict{UInt8, Set{UInt8}}
    @test TracerSparsityDetector(; hessian_pattern_type = H) isa TracerSparsityDetector{TG, HessianTracer{UInt8, Set{UInt8}, Dict{UInt8, Set{UInt8}}, NotShared}}
    @test TracerLocalSparsityDetector(; hessian_pattern_type = H) isa TracerLocalSparsityDetector{TG, HessianTracer{UInt8, Set{UInt8}, Dict{UInt8, Set{UInt8}}, NotShared}}
    @test_throws TypeError TracerSparsityDetector(; hessian_pattern_type = Dict{Int, Set{Float32}})
    @test_throws TypeError TracerLocalSparsityDetector(; hessian_pattern_type = Dict{Int, Set{Float32}})
    @test_throws MethodError TracerSparsityDetector(; hessian_pattern_type = BitSet)
    @test_throws MethodError TracerLocalSparsityDetector(; hessian_pattern_type = BitSet)

    # Set of Tuples
    H = Set{Tuple{UInt32, UInt32}}
    @test TracerSparsityDetector(; hessian_pattern_type = H) isa TracerSparsityDetector{TG, HessianTracer{UInt32, Set{UInt32}, Set{Tuple{UInt32, UInt32}}, NotShared}}
    @test TracerLocalSparsityDetector(; hessian_pattern_type = H) isa TracerLocalSparsityDetector{TG, HessianTracer{UInt32, Set{UInt32}, Set{Tuple{UInt32, UInt32}}, NotShared}}
    @test_throws TypeError TracerSparsityDetector(; hessian_pattern_type = Set{Tuple{Float32, Float32}})
    @test_throws TypeError TracerLocalSparsityDetector(; hessian_pattern_type = Set{Tuple{Float32, Float32}})

    ## Shared
    @test TracerSparsityDetector(; shared_hessian_pattern = true) isa TracerSparsityDetector{TG, HessianTracer{Int64, BitSet, Dict{Int64, BitSet}, Shared}}
    @test TracerLocalSparsityDetector(; shared_hessian_pattern = true) isa TracerLocalSparsityDetector{TG, HessianTracer{Int64, BitSet, Dict{Int64, BitSet}, Shared}}
end
