using Dates: now
using LinearAlgebra
using OptimizationProblems
using SparseArrays
using Test
using SparseConnectivityTracerBenchmarks.Optimization:
    compute_jac_and_hess_sparsity_sct,
    compute_jac_and_hess_sparsity_and_value_jump,
    optimization_problem_names

function compare_patterns(
        truth::AbstractMatrix{<:Real}; sct::AbstractMatrix{Bool}, jump::AbstractMatrix{Bool}
    )
    difference = jump - sct
    if nnz(difference) > 0
        # test that all pattern differences are local zeros in the ground truth
        I, J, _ = findnz(difference)
        coeffs = [truth[i, j] for (i, j) in zip(I, J)]
        @test maximum(abs, coeffs) < 1.0e-7
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

#=
Please look at the warnings displayed at the end.
=#

jac_inconsistencies = []
hess_inconsistencies = []

@testset "$name" for name in optimization_problem_names()
    @info "$(now()) - $name"

    (jac_sparsity_sct, hess_sparsity_sct) = compute_jac_and_hess_sparsity_sct(name)
    ((jac_sparsity_jump, jac), (hess_sparsity_jump, hess)) = compute_jac_and_hess_sparsity_and_value_jump(
        name
    )

    @testset verbose = true "Jacobian comparison" begin
        if jac_sparsity_sct == jac_sparsity_jump
            @test jac_sparsity_sct == jac_sparsity_jump
        else
            @test_broken jac_sparsity_sct == jac_sparsity_jump
            message = compare_patterns(jac; sct = jac_sparsity_sct, jump = jac_sparsity_jump)
            @warn "Inconsistency for Jacobian of $name: $message"
            push!(jac_inconsistencies, (name, message))
        end
    end

    @testset verbose = true "Hessian comparison" begin
        if hess_sparsity_sct == hess_sparsity_jump
            @test hess_sparsity_sct == hess_sparsity_jump
        else
            @test_broken hess_sparsity_sct == hess_sparsity_jump
            message = compare_patterns(hess; sct = hess_sparsity_sct, jump = hess_sparsity_jump)
            @warn "Inconsistency for Hessian of $name: $message"
            push!(hess_inconsistencies, (name, message))
        end
    end
    yield()
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
