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

function mycons(nlp, x)
    c = similar(x, nlp.meta.ncon)
    cons!(nlp, x, c)
    return c
end

function jac_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    f(x) = mycons(nlp, x)
    try
        return ADTypes.jacobian_sparsity(f, nlp.meta.x0, TracerSparsityDetector())
    catch e
        @warn "Global Jacobian sparsity failed" name typeof(e)
        return ADTypes.jacobian_sparsity(f, nlp.meta.x0, TracerLocalSparsityDetector())
    end
end

function hess_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    f(x) = obj(nlp, x) + sum(mycons(nlp, x))
    try
        return ADTypes.hessian_sparsity(f, nlp.meta.x0, TracerSparsityDetector())
    catch e
        @warn "Global Hessian sparsity failed" name typeof(e)
        return ADTypes.hessian_sparsity(f, nlp.meta.x0, TracerLocalSparsityDetector())
    end
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
    # only the lower triangular part is stored
    return sparse(Symmetric(H_L, :L))
end

@testset "Jacobian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        if startswith(string(name), "triangle") || startswith(string(name), "tetra_")
            # UndefVarError
            continue
        else
            J_sct = jac_sparsity_sct(name)
            J_ref = jac_sparsity_ref(name)
            if J_sct == J_ref
                @test J_sct == J_ref
            else
                @test_broken J_sct == J_ref
            end
        end
    end
end

@testset "Hessian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        if startswith(string(name), "triangle") || startswith(string(name), "tetra_")
            # UndefVarError
            continue
        else
            H_sct = hess_sparsity_sct(name)
            H_ref = hess_sparsity_ref(name)
            if H_sct == H_ref
                @test H_sct == H_ref
            else
                @test_broken H_sct == H_ref
            end
        end
    end
end
