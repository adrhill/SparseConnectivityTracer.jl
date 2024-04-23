## Enumerate inputs

"""
    trace_input(JacobianTracer, x)
    trace_input(ConnectivityTracer, x)


Enumerates input indices and constructs the specified type of tracer.
Supports [`JacobianTracer`](@ref) and [`ConnectivityTracer`](@ref).

## Example
```jldoctest
julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];

julia> xt = trace_input(ConnectivityTracer, x)
3-element Vector{ConnectivityTracer}:
 ConnectivityTracer(1,)
 ConnectivityTracer(2,)
 ConnectivityTracer(3,)

julia> yt = f(xt)
3-element Vector{ConnectivityTracer}:
   ConnectivityTracer(1,)
 ConnectivityTracer(1, 2)
   ConnectivityTracer(3,)
```
"""
trace_input(::Type{T}, x) where {T<:AbstractTracer} = trace_input(T, x, 1)
trace_input(::Type{T}, ::Number, i) where {T<:AbstractTracer} = tracer(T, i)
function trace_input(::Type{T}, x::AbstractArray, i) where {T<:AbstractTracer}
    indices = (i - 1) .+ reshape(1:length(x), size(x))
    return tracer.(T, indices)
end

## Construct sparsity pattern matrix
"""
    pattern(f, JacobianTracer, x)

Computes the sparsity pattern of the Jacobian of `y = f(x)`.

    pattern(f, ConnectivityTracer, x)

Enumerates inputs `x` and primal outputs `y = f(x)` and returns sparse matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.

## Example
```jldoctest
julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];

julia> pattern(f, ConnectivityTracer, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 4 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  1
```
"""
function pattern(f, ::Type{T}, x) where {T<:AbstractTracer}
    xt = trace_input(T, x)
    yt = f(xt)
    return _pattern(xt, yt)
end

"""
    pattern(f!, y, JacobianTracer, x)

Computes the sparsity pattern of the Jacobian of `f!(y, x)`.

    pattern(f!, y, ConnectivityTracer, x)

Enumerates inputs `x` and primal outputs `y` after `f!(y, x)` and returns sparse matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.
"""
function pattern(f!, y, ::Type{T}, x) where {T<:AbstractTracer}
    xt = trace_input(T, x)
    yt = similar(y, T)
    f!(yt, xt)
    return _pattern(xt, yt)
end

_pattern(xt::AbstractTracer, yt::Number) = _pattern([xt], [yt])
_pattern(xt::AbstractTracer, yt::AbstractArray{Number}) = _pattern([xt], yt)
_pattern(xt::AbstractArray{<:AbstractTracer}, yt::Number) = _pattern(xt, [yt])
function _pattern(xt::AbstractArray{<:AbstractTracer}, yt::AbstractArray{<:Number})
    return _pattern_to_sparsemat(xt, yt)
end

function _pattern_to_sparsemat(
    xt::AbstractArray{<:AbstractTracer}, yt::AbstractArray{<:Number}
)
    # Construct matrix of size (ouput_dim, input_dim)
    n, m = length(xt), length(yt)
    I = UInt64[]
    J = UInt64[]
    V = Bool[]
    for (i, y) in enumerate(yt)
        if y isa AbstractTracer
            for j in inputs(y)
                push!(I, i)
                push!(J, j)
                push!(V, true)
            end
        end
    end
    return sparse(I, J, V, m, n)
end
