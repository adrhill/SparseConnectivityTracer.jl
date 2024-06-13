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

function jac_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    c = Base.Fix1(myconstraints, nlp)
    x0 = nlp.meta.x0
    return ADTypes.jacobian_sparsity(c, x0, TracerSparsityDetector())
end

function hess_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    L = Base.Fix1(mylagrangian, nlp)
    x0 = nlp.meta.x0
    return ADTypes.hessian_sparsity(L, x0, TracerSparsityDetector())
end

## JuMP

#=
Here we use OptimizationProblems.PureJuMP because JuMP is the ground truth, but we translate with NLPModelsJuMP to easily query the stuff we need. 

https://jso.dev/OptimizationProblems.jl/stable/tutorial/#Problems-in-JuMP-syntax:-PureJuMP
https://jso.dev/NLPModelsJuMP.jl/stable/tutorial/#MathOptNLPModel
=#

function jac_jump(name::Symbol)
    nlp_jump = OptimizationProblems.PureJuMP.eval(name)()
    nlp = NLPModelsJuMP.MathOptNLPModel(nlp_jump)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    x0 = nlp.meta.x0
    I, J = NLPModels.jac_structure(nlp)
    V = NLPModels.jac_coord(nlp, x0)
    jac = sparse(I, J, V, m, n)
    return jac
end

function hess_jump(name::Symbol)
    nlp_jump = PureJuMP.eval(name)()
    nlp = NLPModelsJuMP.MathOptNLPModel(nlp_jump)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    x0 = nlp.meta.x0
    yrand = rand(m)
    I, J = NLPModels.hess_structure(nlp)
    V = NLPModels.hess_coord(nlp, x0, yrand)
    hess_lower = sparse(I, J, V, n, n)
    hess = sparse(Symmetric(hess_lower, :L))
    return hess
end

## Comparison

function compare_patterns(
    ground_truth::AbstractMatrix{<:Real};
    sct::AbstractMatrix{Bool},
    jump::AbstractMatrix{Bool},
)
    difference = jump - sct
    nnz_sct = nnz(sct)
    nnz_jump = nnz(jump)

    @assert size(ground_truth) == size(difference)
    if nnz(difference) > 0
        I, J, _ = findnz(difference)
        coeffs = [ground_truth[i, j] for (i, j) in zip(I, J)]
        @test all(iszero, coeffs)
    end

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

jac_inconsistencies = []

@testset verbose = true "Jacobian comparison" begin
    @testset "$name" for name in NAMES
        @info "$(now()) - Testing Jacobian sparsity for $name"
        Jb_sct = jac_sparsity_sct(name)
        J_jump = jac_jump(name)
        Jb_jump = (!iszero).(J_jump)
        if Jb_sct == Jb_jump
            @test Jb_sct == Jb_jump
        else
            @test_broken Jb_sct == Jb_jump
            message = compare_patterns(J_jump; sct=Jb_sct, jump=Jb_jump)
            push!(jac_inconsistencies, (name, message))
        end
    end
end;

hess_inconsistencies = []

@testset verbose = true "Hessian comparison" begin
    @testset "$name" for name in NAMES
        @info "$(now()) - Testing Hessian sparsity for $name"
        Hb_sct = hess_sparsity_sct(name)
        H_jump = hess_jump(name)
        Hb_jump = (!iszero).(H_jump)
        if Hb_sct == Hb_jump
            @test Hb_sct == Hb_jump
        else
            @test_broken Hb_sct == Hb_jump
            message = compare_patterns(H_jump; sct=Hb_sct, jump=Hb_jump)
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
