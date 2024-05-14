const DEFAULT_VECTOR_TYPE = BitSet
const DEFAULT_MATRIX_TYPE = Set{Tuple{Int,Int}}

## Enumerate inputs
"""
    trace_input(T, x)
    trace_input(T, x)


Enumerates input indices and constructs the specified type `T` of tracer.
Supports [`ConnectivityTracer`](@ref), [`GlobalGradientTracer`](@ref) and [`GlobalHessianTracer`](@ref).
"""
trace_input(::Type{T}, x) where {T<:AbstractTracer} = trace_input(T, x, 1)
trace_input(::Type{T}, ::Number, i) where {T<:AbstractTracer} = tracer(T, i)
function trace_input(::Type{T}, x::AbstractArray, i) where {T<:AbstractTracer}
    indices = (i - 1) .+ reshape(1:length(x), size(x))
    return tracer.(T, indices)
end

## Trace function
function trace_function(::Type{T}, f, x) where {T<:AbstractTracer}
    xt = trace_input(T, x)
    yt = f(xt)
    return xt, yt
end

function trace_function(::Type{T}, f!, y, x) where {T<:AbstractTracer}
    xt = trace_input(T, x)
    yt = similar(y, T)
    f!(yt, xt)
    return xt, yt
end

to_array(x::Number) = [x]
to_array(x::AbstractArray) = x

## Construct sparsity pattern matrix
"""
    connectivity_pattern(f, x)
    connectivity_pattern(f, x, T)

Enumerates inputs `x` and primal outputs `y = f(x)` and returns sparse matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.

The type of index set `S` can be specified as an optional argument and defaults to `BitSet`.

## Example

```jldoctest
julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sign(x[3])];

julia> connectivity_pattern(f, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, Int64} with 4 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  1
```
"""
function connectivity_pattern(f, x, ::Type{C}=DEFAULT_VECTOR_TYPE) where {C}
    xt, yt = trace_function(ConnectivityTracer{C}, f, x)
    return connectivity_pattern_to_mat(to_array(xt), to_array(yt))
end

"""
    connectivity_pattern(f!, y, x)
    connectivity_pattern(f!, y, x, T)

Enumerates inputs `x` and primal outputs `y` after `f!(y, x)` and returns sparse matrix `C` of size `(m, n)`
where `C[i, j]` is true if the compute graph connects the `i`-th entry in `y` to the `j`-th entry in `x`.

The type of index set `S` can be specified as an optional argument and defaults to `BitSet`.
"""
function connectivity_pattern(f!, y, x, ::Type{C}=DEFAULT_VECTOR_TYPE) where {C}
    xt, yt = trace_function(ConnectivityTracer{C}, f!, y, x)
    return connectivity_pattern_to_mat(to_array(xt), to_array(yt))
end

function connectivity_pattern_to_mat(
    xt::AbstractArray{T}, yt::AbstractArray{<:Number}
) where {T<:ConnectivityTracer}
    n, m = length(xt), length(yt)
    I = Int[] # row indices
    J = Int[] # column indices
    V = Bool[]   # values
    for (i, y) in enumerate(yt)
        if y isa T
            for j in y.inputs
                push!(I, i)
                push!(J, j)
                push!(V, true)
            end
        end
    end
    return sparse(I, J, V, m, n)
end

"""
    jacobian_pattern(f, x)
    jacobian_pattern(f, x, T)

Compute the sparsity pattern of the Jacobian of `y = f(x)`.

The type of index set `S` can be specified as an optional argument and defaults to `BitSet`.

## Example

```jldoctest
julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sign(x[3])];

julia> jacobian_pattern(f, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  ⋅
```
"""
function jacobian_pattern(f, x, ::Type{G}=DEFAULT_VECTOR_TYPE) where {G}
    xt, yt = trace_function(GlobalGradientTracer{G}, f, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

"""
    jacobian_pattern(f!, y, x)
    jacobian_pattern(f!, y, x, T)

Compute the sparsity pattern of the Jacobian of `f!(y, x)`.

The type of index set `S` can be specified as an optional argument and defaults to `BitSet`.
"""
function jacobian_pattern(f!, y, x, ::Type{G}=DEFAULT_VECTOR_TYPE) where {G}
    xt, yt = trace_function(GlobalGradientTracer{G}, f!, y, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

function jacobian_pattern_to_mat(
    xt::AbstractArray{T}, yt::AbstractArray{<:Number}
) where {T<:GlobalGradientTracer}
    n, m = length(xt), length(yt)
    I = Int[] # row indices
    J = Int[] # column indices
    V = Bool[]   # values
    for (i, y) in enumerate(yt)
        if y isa T
            for j in y.grad
                push!(I, i)
                push!(J, j)
                push!(V, true)
            end
        end
    end
    return sparse(I, J, V, m, n)
end

"""
    hessian_pattern(f, x)
    hessian_pattern(f, x, T)

Computes the sparsity pattern of the Hessian of a scalar function `y = f(x)`.

The type of index set `S` can be specified as an optional argument and defaults to `BitSet`.

## Example

```jldoctest
julia> x = rand(5);

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4] + 1*x[5];

julia> hessian_pattern(f, x)
5×5 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅

julia> g(x) = f(x) + x[2]^x[5];

julia> hessian_pattern(g, x)
5×5 SparseArrays.SparseMatrixCSC{Bool, Int64} with 7 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  1  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅  1
```
"""
function hessian_pattern(
    f, x, ::Type{G}=DEFAULT_VECTOR_TYPE, ::Type{H}=DEFAULT_MATRIX_TYPE
) where {G,H}
    xt, yt = trace_function(GlobalHessianTracer{G,H}, f, x)
    return hessian_pattern_to_mat(to_array(xt), yt)
end

function hessian_pattern_to_mat(
    xt::AbstractArray{T}, yt::T
) where {G,H<:AbstractSet,T<:GlobalHessianTracer{G,H}}
    # Allocate Hessian matrix
    n = length(xt)
    I = Int[] # row indices
    J = Int[] # column indices
    V = Bool[]   # values

    for (i, j) in yt.hess
        push!(I, i)
        push!(J, j)
        push!(V, true)
    end
    h = sparse(I, J, V, n, n)
    return h
end
