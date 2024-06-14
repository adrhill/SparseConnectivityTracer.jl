using BenchmarkTools
using OptimizationProblems: ADNLPProblems

include("../test/definitions/nlpmodels_definitions.jl")

function optbench(names::Vector{Symbol})
    suite = BenchmarkGroup()
    for name in names
        nlp = ADNLPProblems.eval(name)()
        suite[name]["Jacobian"] = @benchmarkable compute_jac_sparsity_sct($nlp) evals = 1 samples =
            1
        suite[name]["Hessian"] = @benchmarkable compute_hess_sparsity_sct($nlp) evals = 1 samples =
            1
    end
    return suite
end
