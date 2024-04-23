## Enumerate inputs

"""
    trace_input(x)

Enumerates input indices and constructs [`ConnectivityTracer`](@ref)s.

## Example
```jldoctest
julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];

julia> xt = trace_input(x)
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
trace_input(x) = trace_input(x, 1)
trace_input(::Number, i) = connectivitytracer(i)
function trace_input(x::AbstractArray, i)
    indices = (i - 1) .+ reshape(1:length(x), size(x))
    return connectivitytracer.(indices)
end

## Construct connectivity matrix
"""
    connectivity(f, x)

Enumerates inputs `x` and primal outputs `y=f(x)` and returns sparse connectivity matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.

## Example
```jldoctest
julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];

julia> connectivity(f, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 4 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  1
```
"""
function connectivity(f, x)
    xt = trace_input(x)
    yt = f(xt)
    return _connectivity(xt, yt)
end

"""
    connectivity(f!, y, x)

Enumerates inputs `x` and primal outputs `y` after `f!(y, x)` and returns sparse connectivity matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.
"""
function connectivity(f!, y, x)
    xt = trace_input(x)
    yt = similar(y, ConnectivityTracer)
    f!(yt, xt)
    return _connectivity(xt, yt)
end

_connectivity(xt::ConnectivityTracer, yt::Number) = _connectivity([xt], [yt])
_connectivity(xt::ConnectivityTracer, yt::AbstractArray{Number}) = _connectivity([xt], yt)
_connectivity(xt::AbstractArray{ConnectivityTracer}, yt::Number) = _connectivity(xt, [yt])
function _connectivity(xt::AbstractArray{ConnectivityTracer}, yt::AbstractArray{<:Number})
    return connectivity_sparsematrixcsc(xt, yt)
end

function connectivity_sparsematrixcsc(
    xt::AbstractArray{ConnectivityTracer}, yt::AbstractArray{<:Number}
)
    # Construct connectivity matrix of size (ouput_dim, input_dim)
    n, m = length(xt), length(yt)
    I = UInt64[]
    J = UInt64[]
    V = Bool[]
    for (i, y) in enumerate(yt)
        if y isa ConnectivityTracer
            for j in inputs(y)
                push!(I, i)
                push!(J, j)
                push!(V, true)
            end
        end
    end
    return sparse(I, J, V, m, n)
end
