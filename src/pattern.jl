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
trace_input(::Number, i) = tracer(i)
function trace_input(x::AbstractArray, i)
    indices = (i - 1) .+ reshape(1:length(x), size(x))
    return tracer.(indices)
end

## Construct sparsity pattern matrix
"""
    pattern(f, JacobianTracer, x)
    pattern(f, ConnectivityTracer, x)

Enumerates inputs `x` and primal outputs `y=f(x)` and returns sparse matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.

## Example
```jldoctest
julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];

julia> pattern(f, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 4 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  1
```
"""
function pattern(f, x)
    xt = trace_input(x)
    yt = f(xt)
    return _pattern(xt, yt)
end

"""
    pattern(f!, JacobianTracer, y, x)
    pattern(f!, ConnectivityTracer, y, x)

Enumerates inputs `x` and primal outputs `y` after `f!(y, x)` and returns sparse matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.
"""
function pattern(f!, y, x)
    xt = trace_input(x)
    yt = similar(y, ConnectivityTracer)
    f!(yt, xt)
    return _pattern(xt, yt)
end

_pattern(xt::ConnectivityTracer, yt::Number) = _pattern([xt], [yt])
_pattern(xt::ConnectivityTracer, yt::AbstractArray{Number}) = _pattern([xt], yt)
_pattern(xt::AbstractArray{ConnectivityTracer}, yt::Number) = _pattern(xt, [yt])
function _pattern(xt::AbstractArray{ConnectivityTracer}, yt::AbstractArray{<:Number})
    return pattern_sparsematrixcsc(xt, yt)
end

function pattern_sparsematrixcsc(
    xt::AbstractArray{ConnectivityTracer}, yt::AbstractArray{<:Number}
)
    # Construct matrix of size (ouput_dim, input_dim)
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
