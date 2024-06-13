using ADTypes: ADTypes
using SparseConnectivityTracer
import SparseConnectivityTracer as SCT

using ADNLPModels: ADNLPModels
using ForwardDiff: ForwardDiff
using NLPModels: NLPModels, AbstractNLPModel
using NLPModelsJuMP: NLPModelsJuMP
using OptimizationProblems: OptimizationProblems

using Dates: now
using LinearAlgebra
using SparseArrays
using Test

#=
Given an optimization problem `min f(x) s.t. c(x) <= 0`, we study

- the Jacobian of the constraints `c(x)`
- the Hessian of the Lagrangian `L(x,y) = f(x) + yᵀc(x)`

Package ecosystem overview: https://jso.dev/ecosystems/models/

- NLPModels.jl: abstract interface `AbstractNLPModel` for nonlinear optimization problems (with utilities to query objective, constraints, and their derivatives). See API at https://jso.dev/NLPModels.jl/stable/api/

- ADNLPModels.jl: concrete `ADNLPModel <: AbstractNLPModel` created from pure Julia code with autodiff
- NLPModelsJuMP.jl: concrete `MathOptNLPModel <: AbstractNLPModel` converted from a `JuMP.Model`

- OptimizationProblems.jl: suite of benchmark problems available in two formulations:
  - OptimizationProblems.ADNLPProblems: spits out `ADNLPModel`
  - OptimizationProblems.PureJuMP: spits out `JuMP.Model`
=#

NAMES = Symbol.(OptimizationProblems.meta[!, :name])

## SCT

#=
Here we use OptimizationProblems.ADNLPProblems because we need the problems in pure Julia.

https://jso.dev/OptimizationProblems.jl/stable/tutorial/#Problems-in-ADNLPModel-syntax:-ADNLPProblems
=#

function myconstraints(nlp::AbstractNLPModel, x::AbstractVector)
    c = similar(x, nlp.meta.ncon)
    NLPModels.cons!(nlp, x, c)
    return c
end

function mylagrangian(nlp::AbstractNLPModel, x::AbstractVector)
    f = NLPModels.obj(nlp, x)
    c = myconstraints(nlp, x)
    y = randn(length(c))
    L = f + dot(y, c)
    return L
end

function compute_jac_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    c = Base.Fix1(myconstraints, nlp)
    x0 = nlp.meta.x0
    jac_sparsity = ADTypes.jacobian_sparsity(c, x0, TracerSparsityDetector())
    return jac_sparsity
end

function compute_hess_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    L = Base.Fix1(mylagrangian, nlp)
    x0 = nlp.meta.x0
    hess_sparsity = ADTypes.hessian_sparsity(L, x0, TracerSparsityDetector())
    return hess_sparsity
end

## JuMP

#=
Here we use OptimizationProblems.PureJuMP because JuMP is the ground truth, but we translate with NLPModelsJuMP to easily query the stuff we need. 

https://jso.dev/OptimizationProblems.jl/stable/tutorial/#Problems-in-JuMP-syntax:-PureJuMP
https://jso.dev/NLPModelsJuMP.jl/stable/tutorial/#MathOptNLPModel
=#

function compute_jac_sparsity_and_value_jump(name::Symbol)
    nlp_jump = OptimizationProblems.PureJuMP.eval(name)()
    nlp = NLPModelsJuMP.MathOptNLPModel(nlp_jump)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    x0 = nlp.meta.x0
    I, J = NLPModels.jac_structure(nlp)
    V = NLPModels.jac_coord(nlp, x0)
    jac_sparsity = sparse(I, J, ones(Bool, length(I)), m, n)
    jac = sparse(I, J, V, m, n)
    return jac_sparsity, jac
end

function compute_hess_sparsity_and_value_jump(name::Symbol)
    nlp_jump = OptimizationProblems.PureJuMP.eval(name)()
    nlp = NLPModelsJuMP.MathOptNLPModel(nlp_jump)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    x0 = nlp.meta.x0
    yrand = rand(m)
    I, J = NLPModels.hess_structure(nlp)
    V = NLPModels.hess_coord(nlp, x0, yrand)
    hess_sparsity = sparse(Symmetric(sparse(I, J, ones(Bool, length(I)), n, n), :L))
    hess = sparse(Symmetric(sparse(I, J, V, n, n), :L))
    return hess_sparsity, hess
end

## Comparison

function compare_patterns(
    truth::AbstractMatrix{<:Real}; sct::AbstractMatrix{Bool}, jump::AbstractMatrix{Bool}
)
    difference = jump - sct
    if nnz(difference) > 0
        # test that all pattern differences are local zeros in the ground truth
        I, J, _ = findnz(difference)
        coeffs = [truth[i, j] for (i, j) in zip(I, J)]
        @test maximum(abs, coeffs) < 1e-7
    end

    nnz_sct, nnz_jump = nnz(sct), nnz(jump)
    diagonal = (difference == Diagonal(difference)) ? "[diagonal difference only]" : ""
    message = if all(>(0), nonzeros(difference))
        "SCT ($nnz_sct nz) ⊂ JuMP ($nnz_jump nz) $diagonal"
    elseif all(<(0), nonzeros(difference))
        "SCT ($nnz_sct nz) ⊃ JuMP ($nnz_jump nz) $diagonal"
    else
        "SCT ($nnz_sct nz) ≠ JuMP ($nnz_jump nz) $diagonal"
    end
    return message
end

## Actual tests

#=
Please look at the warnings displayed at the end.
=#

jac_inconsistencies = []

@testset verbose = true "Jacobian comparison" begin
    @testset "$name" for name in NAMES
        @info "$(now()) - Testing Jacobian sparsity for $name"
        jac_sparsity_sct = compute_jac_sparsity_sct(name)
        jac_sparsity_jump, jac = compute_jac_sparsity_and_value_jump(name)
        if jac_sparsity_sct == jac_sparsity_jump
            @test jac_sparsity_sct == jac_sparsity_jump
        else
            @test_broken jac_sparsity_sct == jac_sparsity_jump
            message = compare_patterns(jac; sct=jac_sparsity_sct, jump=jac_sparsity_jump)
            @warn "Inconsistency for Jacobian of $name: $message"
            push!(jac_inconsistencies, (name, message))
        end
    end
end;

hess_inconsistencies = []

@testset verbose = true "Hessian comparison" begin
    @testset "$name" for name in NAMES
        @info "$(now()) - Testing Hessian sparsity for $name"
        hess_sparsity_sct = compute_hess_sparsity_sct(name)
        hess_sparsity_jump, hess = compute_hess_sparsity_and_value_jump(name)
        if hess_sparsity_sct == hess_sparsity_jump
            @test hess_sparsity_sct == hess_sparsity_jump
        else
            @test_broken hess_sparsity_sct == hess_sparsity_jump
            message = compare_patterns(hess; sct=hess_sparsity_sct, jump=hess_sparsity_jump)
            @warn "Inconsistency for Hessian of $name: $message"
            push!(hess_inconsistencies, (name, message))
        end
    end
end;

if !isempty(jac_inconsistencies) || !isempty(hess_inconsistencies)
    @warn "Inconsistencies were detected"
    for (name, message) in jac_inconsistencies
        @warn "Inconsistency for Jacobian of $name: $message"
    end
    for (name, message) in hess_inconsistencies
        @warn "Inconsistency for Hessian of $name: $message"
    end
end
