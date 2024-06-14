using ADTypes: ADTypes
using SparseConnectivityTracer
import SparseConnectivityTracer as SCT

using ADNLPModels: ADNLPModels
using NLPModels: NLPModels, AbstractNLPModel
using NLPModelsJuMP: NLPModelsJuMP
using OptimizationProblems: OptimizationProblems

using Dates: now
using LinearAlgebra
using SparseArrays

problem_names() = Symbol.(OptimizationProblems.meta[!, :name])

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

function compute_jac_sparsity_sct(nlp::AbstractNLPModel)
    c = Base.Fix1(myconstraints, nlp)
    x0 = nlp.meta.x0
    jac_sparsity = ADTypes.jacobian_sparsity(c, x0, TracerSparsityDetector())
    return jac_sparsity
end

function compute_jac_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    return compute_jac_sparsity_sct(nlp)
end

function compute_hess_sparsity_sct(nlp::AbstractNLPModel)
    L = Base.Fix1(mylagrangian, nlp)
    x0 = nlp.meta.x0
    hess_sparsity = ADTypes.hessian_sparsity(L, x0, TracerSparsityDetector())
    return hess_sparsity
end

function compute_hess_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    return compute_hess_sparsity_sct(nlp)
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
