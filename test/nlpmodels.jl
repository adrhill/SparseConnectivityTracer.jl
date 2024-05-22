using ADTypes: ADTypes
using SparseConnectivityTracer
import SparseConnectivityTracer as SCT
using OptimizationProblems
using ADNLPModels
using NLPModels
using NLPModelsJuMP
using LinearAlgebra
using SparseArrays
using Test

function jac_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    c(x) = cons(nlp, x)
    return ADTypes.jacobian_sparsity(c, nlp.meta.x0, TracerLocalSparsityDetector())
end

function hess_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    f(x) = obj(nlp, x)
    return ADTypes.hessian_sparsity(f, nlp.meta.x0, TracerLocalSparsityDetector())
end

function jac_sparsity_ref(name::Symbol)
    jump_model = OptimizationProblems.PureJuMP.eval(name)()
    nlp = MathOptNLPModel(jump_model)
    jrows, jcols = jac_structure(nlp)
    nnzj = length(jrows)
    jvals = ones(Bool, nnzj)
    return sparse(jrows, jcols, jvals, nlp.meta.ncon, nlp.meta.nvar)
end

function hess_sparsity_ref(name::Symbol)
    jump_model = OptimizationProblems.PureJuMP.eval(name)()
    nlp = MathOptNLPModel(jump_model)
    hrows, hcols = hess_structure(nlp)
    nnzh = length(hrows)
    hvals = ones(Bool, nnzh)
    H_L = sparse(hrows, hcols, hvals, nlp.meta.nvar, nlp.meta.nvar)
    return sparse(Symmetric(H_L, :L))
end

@testset "Jacobian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        J_sct = jac_sparsity_sct(name)
        J_ref = jac_sparsity_ref(name)
        @test J_sct == J_ref
    end
end

@testset "Hessian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        H_sct = hess_sparsity_sct(name)
        H_ref = hess_sparsity_ref(name)
        @test H_sct == H_ref
    end
end
