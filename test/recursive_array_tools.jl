using RecursiveArrayTools

function f!(du, u)
    du.foo[2] = u.foo[1]
    du.foo[1] = u.foo[2]
    return nothing
end

u = NamedArrayPartition(; foo=rand(5))
du = copy(u)

jacobian_sparsity(f!, du, u, TracerSparsityDetector())