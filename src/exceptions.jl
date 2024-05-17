struct MissingPrimalError <: Exception
    fn::Function
    tracer::AbstractTracer
end

function Base.showerror(io::IO, e::MissingPrimalError)
    println(io, "Function ", e.fn, " requires primal value(s).")
    print(
        io,
        "A dual-number tracer for local sparsity detection can be used via `",
        str_local_pattern_fn(e.tracer),
        "`.",
    )
    return nothing
end

str_pattern_fn(::ConnectivityTracer) = "connectivity_pattern"
str_pattern_fn(::GradientTracer)     = "jacobian_pattern"
str_pattern_fn(::HessianTracer)      = "hessian_pattern"

str_local_pattern_fn(::ConnectivityTracer) = "local_connectivity_pattern"
str_local_pattern_fn(::GradientTracer)     = "local_jacobian_pattern"
str_local_pattern_fn(::HessianTracer)      = "local_hessian_pattern"
