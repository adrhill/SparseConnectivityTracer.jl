const DEFAULT_VECTOR_TYPE = BitSet
const DEFAULT_MATRIX_TYPE = Set{Tuple{Int,Int}}

## Enumerate inputs
"""
    trace_input(T, x)
    trace_input(T, x)


Enumerates input indices and constructs the specified type `T` of tracer.
Supports [`ConnectivityTracer`](@ref), [`GradientTracer`](@ref) and [`HessianTracer`](@ref).
"""
trace_input(::Type{T}, x) where {T<:AbstractTracer} = trace_input(T, x, 1)

function trace_input(::Type{T}, x::Number, i::Integer) where {T<:AbstractTracer}
    return create_tracer(T, x, i)
end
function trace_input(::Type{T}, xs::AbstractArray, i) where {T<:AbstractTracer}
    indices = reshape(1:length(xs), size(xs)) .+ (i - 1)
    return create_tracer.(T, xs, indices)
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
            for j in inputs(y)
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
    xt, yt = trace_function(GradientTracer{G}, f, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

"""
    local_jacobian_pattern(f, x)
    local_jacobian_pattern(f, x, T)

Compute the local sparsity pattern of the Jacobian of `y = f(x)` at `x`.

The type of index set `S` can be specified as an optional argument and defaults to `BitSet`.

## Example

```jldoctest
julia> x = [1.0, 2.0, 3.0];

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, max(x[2],x[3])];

julia> local_jacobian_pattern(f, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, Int64} with 4 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  1
```
"""
function local_jacobian_pattern(f, x, ::Type{G}=DEFAULT_VECTOR_TYPE) where {G}
    D = Dual{eltype(x),GradientTracer{G}}
    xt, yt = trace_function(D, f, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

"""
    jacobian_pattern(f!, y, x)
    jacobian_pattern(f!, y, x, T)

Compute the sparsity pattern of the Jacobian of `f!(y, x)`.

The type of index set `S` can be specified as an optional argument and defaults to `BitSet`.
"""
function jacobian_pattern(f!, y, x, ::Type{G}=DEFAULT_VECTOR_TYPE) where {G}
    xt, yt = trace_function(GradientTracer{G}, f!, y, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

"""
    local_jacobian_pattern(f!, y, x)
    local_jacobian_pattern(f!, y, x, T)

Compute the local sparsity pattern of the Jacobian of `f!(y, x)` at `x`.

The type of index set `S` can be specified as an optional argument and defaults to `BitSet`.
"""
function local_jacobian_pattern(f!, y, x, ::Type{G}=DEFAULT_VECTOR_TYPE) where {G}
    D = Dual{eltype(x),GradientTracer{G}}
    xt, yt = trace_function(D, f!, y, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

function jacobian_pattern_to_mat(
    xt::AbstractArray{TT}, yt::AbstractArray{<:Number}
) where {P,T<:GradientTracer,D<:Dual{P,T},TT<:Union{T,D}}
    n, m = length(xt), length(yt)
    I = Int[] # row indices
    J = Int[] # column indices
    V = Bool[]   # values
    for (i, y) in enumerate(yt)
        if y isa TT
            for j in gradient(y)
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
    xt, yt = trace_function(HessianTracer{G,H}, f, x)
    return hessian_pattern_to_mat(to_array(xt), yt)
end

"""
    local_hessian_pattern(f, x)
    local_hessian_pattern(f, x, T)

Computes the local sparsity pattern of the Hessian of a scalar function `y = f(x)` at `x`.

The type of index set `S` can be specified as an optional argument and defaults to `BitSet`.

## Example

```jldoctest
julia> x = [1.0 3.0 5.0 1.0 2.0];

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4] + x[2] * max(x[1], x[5]);

julia> local_hessian_pattern(f, x)
5×5 SparseArrays.SparseMatrixCSC{Bool, Int64} with 5 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅  ⋅
```
"""
function local_hessian_pattern(
    f, x, ::Type{G}=DEFAULT_VECTOR_TYPE, ::Type{H}=DEFAULT_MATRIX_TYPE
) where {G,H}
    D = Dual{eltype(x),HessianTracer{G,H}}
    xt, yt = trace_function(D, f, x)
    return hessian_pattern_to_mat(to_array(xt), yt)
end

function hessian_pattern_to_mat(
    xt::AbstractArray{TT}, yt::TT
) where {P,T<:HessianTracer,D<:Dual{P,T},TT<:Union{T,D}}

    # Allocate Hessian matrix
    n = length(xt)
    I = Int[] # row indices
    J = Int[] # column indices
    V = Bool[]   # values

    for (i, j) in hessian(yt)
        push!(I, i)
        push!(J, j)
        push!(V, true)
    end
    h = sparse(I, J, V, n, n)
    return h
end
