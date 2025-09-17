module SparseConnectivityTracerChainRulesCoreExt

import ChainRulesCore: ignore_derivatives
using SparseConnectivityTracer: AbstractTracer, Dual, primal, myempty

## Scalar
ignore_derivatives(::T) where {T <: AbstractTracer} = myempty(T)
ignore_derivatives(d::Dual) = primal(d)

## Array
ignore_derivatives(A::AbstractArray{T}) where {T <: AbstractTracer} = ignore_derivatives.(A)
ignore_derivatives(A::AbstractArray{<:Dual}) = primal.(A)

end # module
