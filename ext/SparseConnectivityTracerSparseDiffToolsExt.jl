module SparseConnectivityTracerSparseDiffToolsExt

using SparseConnectivityTracer: connectivity
using SparseDiffTools:
    AbstractSparseADType,
    AbstractSparsityDetection,
    ArrayInterface,
    GreedyD1Color,
    JacPrototypeSparsityDetection,
    SparseDiffTools

Base.@kwdef struct ConnectivityTracerSparsityDetection{
    A<:ArrayInterface.ColoringAlgorithm
} <: AbstractSparsityDetection
    alg::A = GreedyD1Color()
end

function (alg::ConnectivityTracerSparsityDetection)(
    ad::AbstractSparseADType, f, x; fx=nothing, kwargs...
)
    fx = fx === nothing ? similar(f(x)) : dx
    J = connectivity(f, x)
    _alg = JacPrototypeSparsityDetection(J, alg.alg)
    return _alg(ad, f, x; fx, kwargs...)
end

function (alg::ConnectivityTracerSparsityDetection)(
    ad::AbstractSparseADType, f!, fx, x; kwargs...
)
    J = connectivity(f!, fx, x)
    _alg = JacPrototypeSparsityDetection(J, alg.alg)
    return _alg(ad, f!, fx, x; kwargs...)
end

end
