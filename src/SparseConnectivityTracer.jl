module SparseConnectivityTracer

# Input connectivity tracer
struct Tracer <: Number
    inputs::Set{UInt64} # indices of connected, enumerated inputs
end

Tracer(i::Integer) = Tracer(Set{UInt64}(i))
Tracer(a::Tracer, b::Tracer) = Tracer(a.inputs ∪ b.inputs)

# Enumerate inputs
inputtrace(x) = inputtrace(x, 1)
inputtrace(::Number, i) = Tracer(i)
function inputtrace(x::AbstractArray, i)
    indices = (i - 1) .+ reshape(1:length(x), size(x))
    return Tracer.(indices)
end

include("ops.jl")

# Extent core operators
for fn in (:+, :-, :*, :/, :^)
    @eval Base.$fn(a::Tracer, b::Tracer) = Tracer(a, b)
    for T in (:Number,)
        @eval Base.$fn(t::Tracer, ::$T) = t
        @eval Base.$fn(::$T, t::Tracer) = t
    end
end

Base.:^(a::Tracer, b::Tracer) = Tracer(a, b)
for T in (:Number, :Integer, :Rational)
    @eval Base.:^(t::Tracer, ::$T) = t
    @eval Base.:^(::$T, t::Tracer) = t
end
Base.:^(t::Tracer, ::Irrational{:ℯ}) = t
Base.:^(::Irrational{:ℯ}, t::Tracer) = t

# Two-argument functions
for fn in (:div, :fld, :cld)
    @eval Base.$fn(a::Tracer, b::Tracer) = Tracer(a, b)
    @eval Base.$fn(t::Tracer, ::Number) = t
    @eval Base.$fn(::Number, t::Tracer) = t
end

# Single-argument functions
for fn in scalar_operations
    @eval Base.$fn(t::Tracer) = t
end

function connectivity(f, x)
    xt = inputtrace(x)
    yt = f(xt)
    n, m = length(xt), length(yt)

    # Construct connectivity matrix of size (ouput_dim, input_dim)
    C = BitArray(undef, m, n)
    for i in axes(C, 1)
        tracer = yt[i]
        for j in axes(C, 2)
            C[i, j] = j ∈ tracer.inputs
        end
    end
    return C
end

export connectivity

end
