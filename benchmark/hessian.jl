using BenchmarkTools

using ADTypes: hessian_sparsity
using SparseConnectivityTracer

using SparseArrays: sprand

#=
Test cases taken from the article:
> "On efficient Hessian computation using the edge pushing algorithm in Julia"
> https://www.tandfonline.com/doi/full/10.1080/10556788.2018.1480625
=#

function hessbench(method)
    suite = BenchmarkGroup()
    suite["ArrowHead"] = hessbench_arrowhead(method)
    suite["RandomSparsity"] = hessbench_randomsparsity(method)
    return suite
end

## 4.1 Arrow-head structure

struct ArrowHead
    K::Int
end

function (ah::ArrowHead)(x::AbstractVector)
    K = ah.K
    N = length(x)
    return sum(1:N) do i
        s1 = sum(x[i + j] for j in 1:K if (i + j) in eachindex(x); init=zero(eltype(x)))
        s2 = sum((x[i] + x[j])^2 for j in 1:K)
        cos(s1) + s2
    end
end

function hessbench_arrowhead(method)
    suite = BenchmarkGroup()
    # Commented-out cases (N, K) are included in the JuMP paper linked above,
    # but excluded from to accelerate the benchmark suite.
    for (N, K) in [
        ## Table 1
        (200, 16),
        # (400, 16),
        # (800, 16),
        ## Table 2
        (3200, 2),
        # (3200, 4),
        # (3200, 8),
    ]
        x = rand(N)
        f = ArrowHead(K)
        suite["(N=$N, K=$K)"] = @benchmarkable hessian_sparsity($f, $x, $method)
    end
    return suite
end

# 4.2 Random sparsity structure

struct RandomSparsity
    rand_sets::Vector{Vector{Int}}
end

function RandomSparsity(N::Integer, K::Integer)
    rand_sets = [rand(1:N, K) for i in 1:N]
    return RandomSparsity(rand_sets)
end

function (rs::RandomSparsity)(x::AbstractVector)
    return sum(eachindex(x, rs.rand_sets)) do i
        (x[i] - 1)^2 + prod(x[rs.rand_sets[i]])
    end
end

function hessbench_randomsparsity(method)
    suite = BenchmarkGroup()
    # Commented-out cases (N, K) are included in the JuMP paper linked above,
    # but excluded from to accelerate the benchmark suite.
    for (N, K) in [
        ## Table 3
        (400, 2),
        # (400, 4),
        # (400, 8),
        ## Table 4
        (100, 32),
        # (200, 32),
        # (400, 32),
    ]
        x = rand(N)
        f = RandomSparsity(N, K)
        suite["(N=$N, K=$K)"] = @benchmarkable hessian_sparsity($f, $x, $method)
    end
    return suite
end

# 4.3 Logistic regression

# TODO: Add this test case
