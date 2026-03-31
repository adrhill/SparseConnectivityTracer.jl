#= Create empty sets and compute unions of sets and cross-products of sets =#

myempty(::S) where {S <: AbstractSet} = S()
myempty(::Type{S}) where {S <: AbstractSet} = S()
myempty(::D) where {D <: AbstractDict} = D()
myempty(::Type{D}) where {D <: AbstractDict} = D()
seed(::Type{S}, i::Integer; offset::Integer = 0) where {S <: AbstractSet} = S(i)
seed(::Type{S}; offset::Integer = 0) where {S <: AbstractSet} = S()

chunks(::Type{<:AbstractSet{I}}, x) where {I} = (I(1):I(length(x)),) # 1 chunk

myunion!(a::S, b::S) where {S <: AbstractSet} = union!(a, b)
function myunion!(a::D, b::D) where {I <: Integer, S <: AbstractSet{I}, D <: AbstractDict{I, S}}
    for k in keys(b)
        if haskey(a, k)
            union!(a[k], b[k])
        else
            push!(a, k => b[k])
        end
    end
    return a
end

# convert to set of index tuples
tuple_set(s::AbstractSet{Tuple{I, I}}) where {I <: Integer} = s
function tuple_set(d::AbstractDict{I, S}) where {I <: Integer, S <: AbstractSet{I}}
    return Set((k, v) for k in keys(d) for v in d[k])
end

""""
    product(a::S{T}, b::S{T})::S{Tuple{T,T}}

Inner product of set-like inputs `a` and `b`.
"""
function product(a::AbstractSet{I}, b::AbstractSet{I}) where {I <: Integer}
    # Since the Hessian is symmetric, we only have to keep track of index-tuples (i,j) with i≤j.
    return Set((i, j) for i in a, j in b if i <= j)
end

function union_product!(
        hessian::H, gradient_x::G, gradient_y::G
    ) where {I <: Integer, G <: AbstractSet{I}, H <: AbstractSet{Tuple{I, I}}}
    for i in gradient_x
        for j in gradient_y
            if i <= j # symmetric Hessian
                push!(hessian, (i, j))
            end
        end
    end
    return hessian
end

function union_product!(
        hessian::AbstractDict{I, S}, gradient_x::S, gradient_y::S
    ) where {I <: Integer, S <: AbstractSet{I}}
    for i in gradient_x
        if !haskey(hessian, i)
            push!(hessian, i => S())
        end
        for j in gradient_y
            if i <= j # symmetric Hessian
                push!(hessian[i], j)
            end
        end
    end
    return hessian
end

## GPU-friendly set type

bitwidth(::Type{T}) where {T <: Unsigned} = sizeof(T) * 8

"""
    FixedSizeBitSet{I<:Unsigned,N}

Fixed-size counterpart of the native `BitSet`, designed to store a bounded set of indices using the individual bits inside a tuple of integers.

!!! tip
    This set type can be passed as `gradient_pattern_type` to [`TracerSparsityDetector`](@ref) and [`TracerLocalSparsityDetector`](@ref), enabling GPU-friendly Jacobian sparsity detection (but not Hessian).

    If the input dimension is `D` and the integer type `I` is `B` bits wide, Jacobian sparsity detection will require of order `D / (N * B)` function calls, each considering a `N * B`-sized chunk of the input indices.

# Fields

- `buckets::NTuple{N,I}`: each of the `N` integers inside `buckets` corresponds to the presence or absence of `bitwidth(I)` indices.
- `offset::Int`: the `r`-th leftmost bit of `buckets[q]` maps to the index `(q-1) * B + r + offset`
"""
struct FixedSizeBitSet{I <: Unsigned, N} <: AbstractSet{Int}
    buckets::NTuple{N, I}
    offset::Int

    function FixedSizeBitSet{I, N}(
            buckets::NTuple{N, I} = ntuple(Returns(zero(I)), Val(N)); offset::Integer = 0
        ) where {I <: Unsigned, N}
        return new{I, N}(buckets, offset)
    end

    function FixedSizeBitSet{I, N}(k::Integer; offset::Integer = 0) where {I <: Unsigned, N}
        l = k - offset
        B = bitwidth(I)
        if 1 <= l <= N * B
            q, r = (l - 1) ÷ B + 1, (l - 1) % B + 1
            buckets = ntuple(Val(N)) do b
                # 2^(B-r) has a single 1 as its r-th leftmost bit
                ifelse(b == q, one(I) << (B - r), zero(I))
            end
        else
            throw(
                ArgumentError(
                    "`FixedSizeBitSet{$I,$N}` with offset $offset cannot contain index $k, it can only contain indices between $(offset + 1) and $(offset + N * B). Try using a larger unsigned integer size if it is supported by your device (e.g. `UInt128`) or increasing the number `N=$N` of integers in the bit set.",
                ),
            )
        end
        return new{I, N}(buckets, offset)
    end
end

function seed(::Type{FixedSizeBitSet{I, N}}; offset::Integer = 0) where {I, N}
    return FixedSizeBitSet{I, N}(; offset)
end

function seed(::Type{FixedSizeBitSet{I, N}}, i::Integer; offset::Integer = 0) where {I, N}
    return FixedSizeBitSet{I, N}(i; offset)
end

Base.length(s::FixedSizeBitSet) = sum(count_ones, s.buckets)

function Base.union(s1::FixedSizeBitSet{I, N}, s2::FixedSizeBitSet{I, N}) where {I, N}
    if s1.offset != s2.offset
        throw(
            ArgumentError(
                "`FixedSizeBitSet` arguments have incompatible offsets: $(s1.offset) != $(s2.offset)",
            ),
        )
    else
        new_buckets = map(|, s1.buckets, s2.buckets)  # bitwise logical OR
        return FixedSizeBitSet{I, N}(new_buckets; offset = s1.offset)
    end
end

function Base.iterate(s::FixedSizeBitSet{I, N}, args::Vararg{Any, A}) where {I, N, A}
    # warning: very slow, use only for printing and debugging
    (; buckets, offset) = s
    B = bitwidth(I)
    indices = Int[]
    for q in eachindex(buckets)
        bits = bitstring(buckets[q])
        for r in eachindex(bits)
            if bits[r] == '1'
                push!(indices, (q - 1) * B + r + offset)
            end
        end
    end
    return iterate(indices, args...)
end

function chunks(::Type{FixedSizeBitSet{I, N}}, x) where {I, N}
    B = bitwidth(I)
    return index_chunks(1:length(x); size = B * N)
end
