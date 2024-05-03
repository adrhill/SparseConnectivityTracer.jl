using ADTypes
using SparseArrays
using SparseConnectivityTracer
using SparseConnectivityTracer: SortedVector
using Test

@testset "Correctness" begin
    @testset "$T - ($k1, $k2)" for T in (Int32, Int64),
        k1 in (0, 10, 100, 1000),
        k2 in (0, 10, 100, 1000)

        for _ in 1:100
            x = SortedVector(rand(T(1):T(1000), k1); already_sorted=false)
            y = SortedVector(sort(rand(T(1):T(1000), k2)); already_sorted=true)
            z = union(x, y)
            @test eltype(z) == T
            @test issorted(z.data)
            @test Set(z.data) == union(Set(x.data), Set(y.data))
            if k1 > 0 && k2 > 0
                @test z[1] == min(x[1], y[1])
                @test z[end] == max(x[end], y[end])
            end
        end
    end
end;

sd = TracerSparsityDetector(SortedVector{UInt})
@test ADTypes.jacobian_sparsity(diff, rand(10), sd) isa SparseMatrixCSC
@test_broken ADTypes.hessian_sparsity(x -> sum(abs2, diff(x)), rand(10), sd) isa
    SparseMatrixCSC
