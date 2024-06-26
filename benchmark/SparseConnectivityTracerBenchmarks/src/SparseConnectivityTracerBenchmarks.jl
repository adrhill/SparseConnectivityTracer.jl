module SparseConnectivityTracerBenchmarks

module ODE
    include("brusselator.jl")
    export Brusselator!, brusselator_2d_loop!
end

module Optimization
    using ADTypes: ADTypes
    using SparseConnectivityTracer
    import SparseConnectivityTracer as SCT

    using ADNLPModels: ADNLPModels
    using NLPModels: NLPModels, AbstractNLPModel
    using NLPModelsJuMP: NLPModelsJuMP
    using OptimizationProblems: OptimizationProblems

    using LinearAlgebra
    using SparseArrays

    include("nlpmodels.jl")
    export optimization_problem_names
    export compute_jac_sparsity_sct, compute_hess_sparsity_sct
    export compute_jac_and_hess_sparsity_sct, compute_jac_and_hess_sparsity_and_value_jump
end

end # module SparseConnectivityTracerBenchmarks
