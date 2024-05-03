const DEFAULT_SET_TYPE = BitSet

## Enumerate inputs
"""
    trace_input(T, x)
    trace_input(T, x)


Enumerates input indices and constructs the specified type `T` of tracer.
Supports [`ConnectivityTracer`](@ref), [`JacobianTracer`](@ref) and [`HessianTracer`](@ref).
"""
trace_input(::Type{T}, x) where {T<:AbstractTracer} = trace_input(T, x, 1)
trace_input(::Type{T}, ::Number, i) where {T<:AbstractTracer} = tracer(T, i)
function trace_input(::Type{T}, x::AbstractArray, i) where {T<:AbstractTracer}
    indices = (i - 1) .+ reshape(1:length(x), size(x))
    return tracer.(T, indices)
end

## Construct sparsity pattern matrix
"""
    connectivity_pattern(f, x)
    connectivity_pattern(f, x, T)

Enumerates inputs `x` and primal outputs `y = f(x)` and returns sparse matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.

The type of index set `T<:AbstractSet{<:Integer}` can be specified as an optional argument and defaults to `BitSet`.

## Example

```jldoctest
julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sign(x[3])];

julia> connectivity_pattern(f, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 4 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  1
```
"""
connectivity_pattern(f, x, settype::Type{S}=DEFAULT_SET_TYPE) where {S} =
    pattern(f, ConnectivityTracer{S}, x)

"""
    connectivity_pattern(f!, y, x)
    connectivity_pattern(f!, y, x, T)

Enumerates inputs `x` and primal outputs `y` after `f!(y, x)` and returns sparse matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.

The type of index set `T<:AbstractSet{<:Integer}` can be specified as an optional argument and defaults to `BitSet`.
"""
function connectivity_pattern(f!, y, x, ::Type{S}=DEFAULT_SET_TYPE) where {S}
    return pattern(f!, y, ConnectivityTracer{S}, x)
end

"""
    jacobian_pattern(f, x)
    jacobian_pattern(f, x, T)

Compute the sparsity pattern of the Jacobian of `y = f(x)`.

The type of index set `T<:AbstractSet{<:Integer}` can be specified as an optional argument and defaults to `BitSet`.

## Example

```jldoctest
julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sign(x[3])];

julia> jacobian_pattern(f, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 3 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  ⋅
```
"""
function jacobian_pattern(f, x, ::Type{S}=DEFAULT_SET_TYPE) where {S}
    return pattern(f, JacobianTracer{S}, x)
end

"""
    jacobian_pattern(f!, y, x)
    jacobian_pattern(f!, y, x, T)

Compute the sparsity pattern of the Jacobian of `f!(y, x)`.

The type of index set `T<:AbstractSet{<:Integer}` can be specified as an optional argument and defaults to `BitSet`.
"""
function jacobian_pattern(f!, y, x, ::Type{S}=DEFAULT_SET_TYPE) where {S}
    return pattern(f!, y, JacobianTracer{S}, x)
end

"""
    hessian_pattern(f, x)
    hessian_pattern(f, x, T)

Computes the sparsity pattern of the Hessian of a scalar function `y = f(x)`.

The type of index set `T<:AbstractSet{<:Integer}` can be specified as an optional argument and defaults to `BitSet`.

## Example

```jldoctest
julia> x = rand(5);

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4] + 1*x[5];

julia> hessian_pattern(f, x)
5×5 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 3 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅

julia> g(x) = f(x) + x[2]^x[5];

julia> hessian_pattern(g, x)
5×5 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 7 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  1  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅  1
```
"""
function hessian_pattern(f, x, ::Type{S}=DEFAULT_SET_TYPE) where {S}
    return pattern(f, HessianTracer{S}, x)
end

function pattern(f, ::Type{T}, x) where {T<:AbstractTracer}
    xt = trace_input(T, x)
    yt = f(xt)
    return _pattern(xt, yt)
end

function pattern(f!, y, ::Type{T}, x) where {T<:AbstractTracer}
    xt = trace_input(T, x)
    yt = similar(y, T)
    f!(yt, xt)
    return _pattern(xt, yt)
end

_pattern(xt::AbstractTracer, yt::Number) = _pattern([xt], [yt])
_pattern(xt::AbstractTracer, yt::AbstractArray{<:Number}) = _pattern([xt], yt)
_pattern(xt::AbstractArray{<:AbstractTracer}, yt::Number) = _pattern(xt, [yt])
function _pattern(xt::AbstractArray{<:AbstractTracer}, yt::AbstractArray{<:Number})
    return _pattern_to_sparsemat(xt, yt)
end

function _pattern_to_sparsemat(
    xt::AbstractArray{T}, yt::AbstractArray{<:Number}
) where {T<:AbstractTracer}
    # Construct matrix of size (ouput_dim, input_dim)
    n, m = length(xt), length(yt)
    I = UInt64[] # row indices
    J = UInt64[] # column indices
    V = Bool[]   # values
    for (i, y) in enumerate(yt)
        if y isa T
            for j in inputs(y)
                push!(I, i)
                push!(J, j)
                push!(V, true)
            end
        end
    end
    return sparse(I, J, V, m, n)
end

function _pattern_to_sparsemat(
    xt::AbstractArray{HessianTracer{S}}, yt::AbstractArray{HessianTracer{S}}
) where {S}
    length(yt) != 1 && error("pattern(f, HessianTracer, x) expects scalar output y=f(x).")
    y = only(yt)

    # Allocate Hessian matrix
    n = length(xt)
    I = UInt64[] # row indices
    J = UInt64[] # column indices
    V = Bool[]   # values

    for i in keys(y.inputs)
        for j in y.inputs[i]
            push!(I, i)
            push!(J, j)
            push!(V, true)
        end
    end
    H = sparse(I, J, V, n, n)
    return H
end
