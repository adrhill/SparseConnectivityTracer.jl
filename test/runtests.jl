using SparseConnectivityTracer
using Test
using Aqua
using JET

@testset "SparseConnectivityTracer.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SparseConnectivityTracer)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(SparseConnectivityTracer; target_defined_modules = true)
    end
    # Write your tests here.
end
