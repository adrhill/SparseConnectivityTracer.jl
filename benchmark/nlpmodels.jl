using BenchmarkTools
using OptimizationProblems: ADNLPProblems
using SparseConnectivityTracerBenchmarks.Optimization:
    compute_jac_sparsity_sct, compute_hess_sparsity_sct

function optbench(names::Vector{Symbol})
    suite = BenchmarkGroup()
    for name in names
        nlp = ADNLPProblems.eval(name)()
        suite[name]["Jacobian"] = @benchmarkable compute_jac_sparsity_sct($nlp)
        suite[name]["Hessian"] = @benchmarkable compute_hess_sparsity_sct($nlp)
    end
    return suite
end
