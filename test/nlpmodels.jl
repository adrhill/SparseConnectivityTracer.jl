using ADTypes: ADTypes
using LinearAlgebra
using SparseArrays
using SparseConnectivityTracer
import SparseConnectivityTracer as SCT
using Test

using Pkg
Pkg.add([
    "ADNLPModels", "ForwardDiff", "OptimizationProblems", "NLPModels", "NLPModelsJuMP"
])

using ADNLPModels
using ForwardDiff: ForwardDiff
using NLPModels
using NLPModelsJuMP
using OptimizationProblems

## ForwardDiff reference

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
    din[i] = 1
    dout = zero(x)
    dout[j] = 1
    return second_directional_derivative(f, x, din, dout)
end

## Function wrappers

function mycons(nlp, x)
    c = similar(x, nlp.meta.ncon)
    cons!(nlp, x, c)
    return c
end

function mylag(nlp, x)
    o = obj(nlp, x)
    c = mycons(nlp, x)
    λ = randn(length(c))
    return o + dot(λ, c)
end

## Jacobian sparsity

function jac_coeff(name::Symbol, i::Integer, j::Integer)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    f = Base.Fix1(mycons, nlp)
    x = nlp.meta.x0
    return jac_coeff(f, x, i, j)
end

function jac_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    f = Base.Fix1(mycons, nlp)
    x = nlp.meta.x0
    return ADTypes.jacobian_sparsity(f, x, TracerSparsityDetector())
end

function jac_sparsity_jump(name::Symbol)
    jump_model = OptimizationProblems.PureJuMP.eval(name)()
    nlp = MathOptNLPModel(jump_model)
    jrows, jcols = jac_structure(nlp)
    nnzj = length(jrows)
    jvals = ones(Bool, nnzj)
    return sparse(jrows, jcols, jvals, nlp.meta.ncon, nlp.meta.nvar)
end

## Hessian sparsity

function hess_coeff(name::Symbol, i::Integer, j::Integer)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    f = Base.Fix1(mylag, nlp)
    x = nlp.meta.x0
    return hess_coeff(f, x, i, j)
end

function hess_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    f = Base.Fix1(mylag, nlp)
    x = nlp.meta.x0
    return ADTypes.hessian_sparsity(f, x, TracerSparsityDetector())
end

function hess_sparsity_jump(name::Symbol)
    jump_model = OptimizationProblems.PureJuMP.eval(name)()
    nlp = MathOptNLPModel(jump_model)
    hrows, hcols = hess_structure(nlp)
    nnzh = length(hrows)
    hvals = ones(Bool, nnzh)
    H_L = sparse(hrows, hcols, hvals, nlp.meta.nvar, nlp.meta.nvar)
    # only the lower triangular part is stored
    return sparse(Symmetric(H_L, :L))
end

## Comparison

function compare_patterns(; sct, jump)
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
