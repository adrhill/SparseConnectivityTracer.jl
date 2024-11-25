#= Parse tracers from function evaluation in `src/trace_functions.jl` into an output matrix. =#

#==========#
# Jacobian #
#==========#

function jacobian_pattern_to_mat(
    xt::AbstractArray{T}, yt::AbstractArray{<:Real}
) where {T<:GradientTracer}
    n, m = length(xt), length(yt)
    I = Int[] # row indices
    J = Int[] # column indices
    V = Bool[]   # values
    for (i, y) in enumerate(yt)
        if y isa T && !isemptytracer(y)
            for j in gradient(y)
                push!(I, i)
                push!(J, j)
                push!(V, true)
            end
        end
    end
    return sparse(I, J, V, m, n)
end

function jacobian_pattern_to_mat(
    xt::AbstractArray{D}, yt::AbstractArray{<:Real}
) where {P,T<:GradientTracer,D<:Dual{P,T}}
    return jacobian_pattern_to_mat(tracer.(xt), _tracer_or_number.(yt))
end

#=========#
# Hessian #
#=========#

function hessian_pattern_to_mat(xt::AbstractArray{T}, yt::T) where {T<:HessianTracer}
    n = length(xt)
    I = Int[] # row indices
    J = Int[] # column indices
    V = Bool[]   # values

    if !isemptytracer(yt)
        for (i, j) in tuple_set(hessian(yt))
            push!(I, i)
            push!(J, j)
            push!(V, true)
            # TODO: return `Symmetric` instead on next breaking release
            push!(I, j)
            push!(J, i)
            push!(V, true)
        end
    end
    h = sparse(I, J, V, n, n)
    return h
end

function hessian_pattern_to_mat(
    xt::AbstractArray{D1}, yt::D2
) where {P1,P2,T<:HessianTracer,D1<:Dual{P1,T},D2<:Dual{P2,T}}
    return hessian_pattern_to_mat(tracer.(xt), tracer(yt))
end

function hessian_pattern_to_mat(xt::AbstractArray{T}, yt::Number) where {T<:HessianTracer}
    return hessian_pattern_to_mat(xt, myempty(T))
end

function hessian_pattern_to_mat(
    xt::AbstractArray{D1}, yt::Number
) where {P1,T<:HessianTracer,D1<:Dual{P1,T}}
    return hessian_pattern_to_mat(tracer.(xt), myempty(T))
end
