struct MissingPrimalError <: Exception
    fn::Function
    tracer::AbstractTracer
end

function Base.showerror(io::IO, e::MissingPrimalError)
    println(io, "Function ", e.fn, " requires primal value(s).")
    print(
        io,
        "A dual-number tracer for local sparsity detection can be used via `TracerLocalSparsityDetector`.",
    )
    return nothing
end
