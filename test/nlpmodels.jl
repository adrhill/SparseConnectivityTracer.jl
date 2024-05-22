using ADTypes: ADTypes
using SparseConnectivityTracer
using OptimizationProblems
using ADNLPModels
using NLPModels
using NLPModelsJuMP
using SparseArrays
using Test

function jac_sparsity_sct(nlp::ADNLPModel)
    c(x) = cons(nlp, x)
    return ADTypes.jacobian_sparsity(c, nlp.meta.x0, TracerLocalSparsityDetector())
end

function hess_sparsity_sct(nlp::ADNLPModel)
    f(x) = obj(nlp, x)
    return ADTypes.hessian_sparsity(f, nlp.meta.x0, TracerLocalSparsityDetector())
end

function jac_sparsity_ref(nlp::MathOptNLPModel)
    jrows, jcols = jac_structure(nlp)
    nnzj = length(jrows)
    jvals = ones(Bool, nnzj)
    return sparse(jrows, jcols, jvals, nlp.meta.ncon, nlp.meta.nvar)
end

function hess_sparsity_ref(nlp::MathOptNLPModel)
    hrows, hcols = hess_structure(nlp)
    nnzh = length(hrows)
    hvals = ones(Bool, nnzh)
    return sparse(hrows, hcols, hvals, nlp.meta.nvar, nlp.meta.nvar)
end

@testset "Jacobian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        nlp = OptimizationProblems.ADNLPProblems.eval(name)()
        jump_model2 = OptimizationProblems.PureJuMP.eval(name)()
        nlp2 = MathOptNLPModel(jump_model2)

        J_sct = jac_sparsity_sct(nlp)
        J_ref = jac_sparsity_ref(nlp2)
        @test J_sct == J_ref
    end
end

@testset "Hessian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        nlp = OptimizationProblems.ADNLPProblems.eval(name)()
        jump_model2 = OptimizationProblems.PureJuMP.eval(name)()
        nlp2 = MathOptNLPModel(jump_model2)

        H_sct = hess_sparsity_sct(nlp)
        H_ref = hess_sparsity_ref(nlp2)
        @test H_sct == H_ref
    end
end
