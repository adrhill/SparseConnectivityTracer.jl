using BenchmarkTools
using OptimizationProblems: ADNLPProblems

include("../test/definitions/nlpmodels_definitions.jl")

function jacbench_opt()
    suite = BenchmarkGroup()
    for name in problem_names()
        nlp = ADNLPProblems.eval(name)()
        suite[name] = @benchmarkable compute_jac_sparsity($nlp) evals = 1 samples = 1
    end
    return suite
end

function hessbench_opt()
    suite = BenchmarkGroup()
    for name in problem_names()
        nlp = ADNLPProblems.eval(name)()
        suite[name] = @benchmarkable compute_hess_sparsity($nlp) evals = 1 samples = 1
    end
    return suite
end
