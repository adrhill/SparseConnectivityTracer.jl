
enumerate_tracers(x) = enumerate_tracers(x, 1)
enumerate_tracers(::Number, i) = Tracer(i)
function enumerate_tracers(x::AbstractArray, i)
    indices = (i - 1) .+ reshape(1:length(x), size(x))
    return Tracer.(indices)
end
