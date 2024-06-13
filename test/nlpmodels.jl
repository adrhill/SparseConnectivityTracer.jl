using ADTypes: ADTypes
using SparseConnectivityTracer
import SparseConnectivityTracer as SCT

using ADNLPModels: ADNLPModels
using ForwardDiff: ForwardDiff
using JuMP: JuMP
using NLPModels: NLPModels, AbstractNLPModel
using OptimizationProblems: OptimizationProblems, ADNLPProblems, PureJuMP
import MathOptInterface as MOI

using Dates: now
using LinearAlgebra
using SparseArrays
using Test

## Function wrappers

#=
We are interested in:

- the Jacobian of the nonlinear constraints c_nln(x) (trivial Jacobian for the linear ones)
- the Hessian of the Lagrangian L(x,λ) = y(x) + λᵀc(x) (where we include all constraints)

We cannot use `NLPModels.cons` directly because it falls back on `NLPModels.cons!` which tries to fill a Vector{Float64} with tracers.
=#

function mynonlinearconstraints(nlp::AbstractNLPModel, x::AbstractVector)
    c_nln = similar(x, nlp.meta.nnln)
    if !isempty(c_nln)
        NLPModels.cons_nln!(nlp, x, c_nln)
    end
    return c_nln
end

function myconstraints(nlp::AbstractNLPModel, x::AbstractVector)
    c = similar(x, nlp.meta.ncon)
    NLPModels.cons!(nlp, x, c)
    return c
end

function mylagrangian(nlp::AbstractNLPModel, x::AbstractVector)
    y = NLPModels.obj(nlp, x)
    c = myconstraints(nlp, x)
    λ = randn(length(c))
    L = y + dot(λ, c)
    return L
end

## ForwardDiff

#=
We use ForwardDiff to check correctness whenever SCT and JuMP disagree.
=#

function directional_derivative(f, x::AbstractVector, d::AbstractVector)
    return ForwardDiff.derivative(t -> f(x + t * d), zero(eltype(x)))
end

function second_directional_derivative(
    f, x::AbstractVector, din::AbstractVector, dout::AbstractVector
)
    f_din(x) = directional_derivative(f, x, din)
    return directional_derivative(f_din, x, dout)
end

function jac_coeff(f, x::AbstractVector, i::Integer, j::Integer)
    d = zero(x)
    d[j] = 1
    return directional_derivative(f, x, d)[i]
end

function hess_coeff(f, x::AbstractVector, i::Integer, j::Integer)
    din = zero(x)
    dout = zero(x)
    din[i] = 1
    dout[j] = 1
    return second_directional_derivative(f, x, din, dout)
end

function jac_coeff(name::Symbol, i::Integer, j::Integer)
    nlp = ADNLPProblems.eval(name)()
    f = Base.Fix1(mynonlinearconstraints, nlp)
    x0 = nlp.meta.x0
    return jac_coeff(f, x0, i, j)
end

function hess_coeff(name::Symbol, i::Integer, j::Integer)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    f = Base.Fix1(mylagrangian, nlp)
    x0 = nlp.meta.x0
    return hess_coeff(f, x0, i, j)
end

## SCT

#=
Here we use OptimizationProblems.ADNLPProblems because we need the problems in pure Julia code.
https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/querying_hessians/#Create-a-nonlinear-model
https://jso.dev/OptimizationProblems.jl/stable/tutorial/#Problems-in-ADNLPModel-syntax:-ADNLPProblems
=#

function jac_sparsity_sct(name::Symbol)
    nlp = ADNLPProblems.eval(name)()
    f = Base.Fix1(mynonlinearconstraints, nlp)
    x0 = nlp.meta.x0
    return ADTypes.jacobian_sparsity(f, x0, TracerSparsityDetector())
end

function hess_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    f = Base.Fix1(mylagrangian, nlp)
    x0 = nlp.meta.x0
    return ADTypes.hessian_sparsity(f, x0, TracerSparsityDetector())
end

## JuMP

#=
Here we use OptimizationProblems.PureJuMP because we need the problems in algebraic form.
https://jso.dev/OptimizationProblems.jl/stable/tutorial/#Problems-in-JuMP-syntax:-PureJuMP

We then retrieve the Hessian sparsity pattern with MathOptInterface.
https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/querying_hessians/#Create-a-nonlinear-model
=#

function jump_to_moi_evaluator(model::JuMP.Model)
    rows = Any[]
    moi_model = MOI.Nonlinear.Model()
    for (F, S) in JuMP.list_of_constraint_types(model)
        if F <: JuMP.VariableRef
            continue  # Skip variable bounds
        end
        for ci in JuMP.all_constraints(model, F, S)
            push!(rows, ci)
            object = JuMP.constraint_object(ci)
            MOI.Nonlinear.add_constraint(moi_model, object.func, object.set)
        end
    end
    MOI.Nonlinear.set_objective(moi_model, JuMP.objective_function(model))

    evaluator = MOI.Nonlinear.Evaluator(
        moi_model, MOI.Nonlinear.SparseReverseMode(), JuMP.index.(JuMP.all_variables(model))
    )
    MOI.initialize(evaluator, [:Hess])
    return evaluator
end

function jac_sparsity_jump(name::Symbol)
    model = PureJuMP.eval(name)()
    evaluator = jump_to_moi_evaluator(model)
    jacobian_IJ = MOI.jacobian_structure(evaluator)  # only nonlinear constraints
    I, J, V = first.(jacobian_IJ), last.(jacobian_IJ), ones(Bool, length(jacobian_IJ))
    return sparse(I, J, V)
end

function hess_sparsity_jump(name::Symbol)
    model = PureJuMP.eval(name)()
    @show model
    evaluator = jump_to_moi_evaluator(model)
    hessian_IJ = MOI.hessian_lagrangian_structure(evaluator)
    I, J, V = first.(hessian_IJ), last.(hessian_IJ), ones(Bool, length(hessian_IJ))
    n = JuMP.num_variables(model)
    hessian_lower = sparse(I, J, V, n, n)
    return sparse(Symmetric(hessian_lower, :L))
end

## Comparison

function compare_patterns(; sct::AbstractMatrix{Bool}, jump::AbstractMatrix{Bool})
    A_diff = jump - sct
    nnz_sct = nnz(sct)
    nnz_jump = nnz(jump)

    diagonal = if A_diff == Diagonal(A_diff)
        "[diagonal difference only]"
    else
        ""
    end
    message = if all(>(0), nonzeros(A_diff))
        "SCT ($nnz_sct nz) ⊂ JuMP ($nnz_jump nz) $diagonal"
    elseif all(<(0), nonzeros(A_diff))
        "SCT ($nnz_sct nz) ⊃ JuMP ($nnz_jump nz) $diagonal"
    else
        "SCT ($nnz_sct nz) ≠ JuMP ($nnz_jump nz) $diagonal"
    end
    return message
end

## Actual tests

@testset verbose = true "ForwardDiff reference" begin
    f(x) = sin.(x) .* cos.(reverse(x)) .* exp(x[1]) .* log(x[end])
    g(x) = sum(f(x))
    x = rand(6)
    @testset "Jacobian" begin
        J = ForwardDiff.jacobian(f, x)
        for i in axes(J, 1), j in axes(J, 2)
            @test J[i, j] == jac_coeff(f, x, i, j)
        end
    end
    @testset "Hessian" begin
        H = ForwardDiff.hessian(g, x)
        for i in axes(H, 1), j in axes(H, 2)
            @test H[i, j] == hess_coeff(g, x, i, j)
        end
    end
end;

jac_inconsistencies = []

@testset verbose = true "Jacobian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        @info "Testing Jacobian sparsity for $name"
        J_sct = jac_sparsity_sct(name)
        J_jump = jac_sparsity_jump(name)
        if J_sct == J_jump
            @test J_sct == J_jump
        else
            @test_broken J_sct == J_jump
            J_diff = J_jump - J_sct
            message = compare_patterns(; sct=J_sct, jump=J_jump)
            # @warn "Inconsistency for Jacobian of $name: $message"
            push!(jac_inconsistencies, (name, message))
            @test all(zip(findnz(J_diff)...)) do (i, j, _)
                iszero(jac_coeff(name, i, j))
            end
        end
    end
end;

hess_inconsistencies = []

@testset verbose = true "Hessian comparison" begin
    @testset "$name" for name in Symbol.(OptimizationProblems.meta[!, :name])
        @info "Testing Hessian sparsity for $name"
        H_sct = hess_sparsity_sct(name)
        H_jump = hess_sparsity_jump(name)
        if H_sct == H_jump
            @test H_sct == H_jump
        else
            @test_broken H_sct == H_jump
            message = compare_patterns(; sct=H_sct, jump=H_jump)
            # @warn "Inconsistency for Hessian of $name: $message"
            push!(hess_inconsistencies, (name, message))
            H_diff = H_jump - H_sct
            @test all(zip(findnz(H_diff)...)) do (i, j, _)
                iszero(hess_coeff(name, i, j))
            end
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
