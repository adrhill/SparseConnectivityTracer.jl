module SparseConnectivityTracerLinearAlgebraExt

import LinearAlgebra
import SparseConnectivityTracer as SCT

@inline function LinearAlgebra.axpy!(α::SCT.AbstractTracer, x::AbstractArray, y::AbstractArray)
    @. y += α * x
    return y
end

@inline function LinearAlgebra.axpby!(α::SCT.AbstractTracer, x::AbstractArray, β, y::AbstractArray)
    @. y = α * x + β * y
    return y
end

@inline function LinearAlgebra.axpby!(α, x::AbstractArray, β::SCT.AbstractTracer, y::AbstractArray)
    @. y = α * x + β * y
    return y
end

end
