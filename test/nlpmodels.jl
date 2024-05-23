using Pkg
# TODO: remove once a new release > 0.7.3 has been tagged
Pkg.add(; url="https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl")

using ADNLPModels
using ADTypes: ADTypes
using LinearAlgebra
using NLPModels
using NLPModelsJuMP
using OptimizationProblems
using SparseArrays
using SparseConnectivityTracer
import SparseConnectivityTracer as SCT
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

@testset verbose = true "Jacobian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        @info "Testing Jacobian sparsity for $name"
        J_sct = jac_sparsity_sct(name)
        J_ref = jac_sparsity_ref(name)
        if name == :lincon
            # we have two more nonzeros but their stored values are 0.0 in the Hessian
            @test_broken J_sct == J_ref
        else
            @test J_sct == J_ref
        end
    end
end;

@testset verbose = true "Hessian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        @info "Testing Hessian sparsity for $name"
        if startswith(string(name), "tetra_")
            # TODO: investigate
            @warn "Skipping $name because it takes too long"
            @test_broken false
        else
            H_sct = hess_sparsity_sct(name)
            H_ref = hess_sparsity_ref(name)
            H_diff = H_ref - H_sct
            # usually the difference is on the diagonal and ref has more nonzeros than SCT
            if name in (:britgas, :channel, :hs114, :marine)
                # TODO: investigate
                @test_broken H_diff == Diagonal(H_diff) && all(H_diff .>= 0)
            else
                @test H_diff == Diagonal(H_diff) && all(H_diff .>= 0)
            end
        end
    end
end;
