# Enumerate inputs
trace(x) = trace(x, 1)
trace(::Number, i) = tracer(i)
function trace(x::AbstractArray, i)
    indices = (i - 1) .+ reshape(1:length(x), size(x))
    return tracer.(indices)
end

istracer(x) = false
istracer(x::Tracer) = true
istracer(x::AbstractArray{Tracer}) = true

# Construct connectivity matrix
function connectivity(f::Function, x)
    xt = trace(x)
    yt = f(xt)
    return _connectivity(xt, yt)
end

_connectivity(xt::Tracer, yt::Number)                = _connectivity([xt], [yt])
_connectivity(xt::Tracer, yt::AbstractArray{Number}) = _connectivity([xt], yt)
_connectivity(xt::AbstractArray{Tracer}, yt::Number) = _connectivity(xt, [yt])

function _connectivity(xt::AbstractArray{Tracer}, yt::AbstractArray{<:Number})
    # Construct connectivity matrix of size (ouput_dim, input_dim)
    n, m = length(xt), length(yt)
    C = BitArray(undef, m, n)
    for i in axes(C, 1)
        if yt[i] isa Tracer
            for j in axes(C, 2)
                C[i, j] = j ∈ yt[i].inputs
            end
        else
            C[i, :] .= 0
        end
    end
    return C
end
