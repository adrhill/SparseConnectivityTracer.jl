using ADTypes
using SparseArrays
using SparseConnectivityTracer
using SparseConnectivityTracer: SortedVector
using Test

@testset "Correctness" begin
    @testset "$T - ($k1, $k2)" for T in (Int32, Int64),
        k1 in (0, 10, 100, 1000),
        k2 in (0, 10, 100, 1000)

        @test all(1:100) do _
            x = SortedVector{T}(rand(T(1):T(1000), k1); sorted=false)
            y = SortedVector{T}(sort(rand(T(1):T(1000), k2)); sorted=true)
            z = union(x, y)
            eltype(z) == T || return false
            issorted(z.data) || return false
            Set(z.data) == union(Set(x.data), Set(y.data)) || return false
            if k1 > 0 && k2 > 0
                xc = collect(x)
                yc = collect(y)
                zc = collect(z)
                zc[1] == min(xc[1], yc[1]) || return false
                zc[end] == max(xc[end], yc[end]) || return false
            end
            return true
        end
    end
end;
