## Operator definitions

# We use a system of letters to categorize operators:
#   z: first- and second-order derivatives (FOD, SOD) are zero
#   f: FOD ∂f/∂x is non-zero, SOD ∂²f/∂x² is zero 
#   s: FOD ∂f/∂x is non-zero, SOD ∂²f/∂x² is non-zero
#   c: Cross-derivative ∂²f/∂x∂y is non-zero

#! format: off

##=================================#
# Operators for functions f: ℝ → ℝ #
#==================================#
function is_firstder_zero_global end
function is_seconder_zero_global end

# Fallbacks for local derivatives:
is_firstder_zero_local(f, x) = is_firstder_zero_global(f)
is_seconder_zero_local(f, x) = is_seconder_zero_global(f)

# ops_1_to_1_s: 
# ∂f/∂x   != 0
# ∂²f/∂x² != 0
ops_1_to_1_s = (
    # trigonometric functions
    :cos, :cosd, :cosh, :cospi, :cosc, 
    :sin, :sind, :sinh, :sinpi, :sinc, 
    :tan, :tand, :tanh,
    # reciprocal trigonometric functions
    :csc, :cscd, :csch, 
    :sec, :secd, :sech, 
    :cot, :cotd, :coth,
    # inverse trigonometric functions
    :acos, :acosd, :acosh, 
    :asin, :asind, :asinh, 
    :atan, :atand, :atanh, 
    :asec, :asech, 
    :acsc, :acsch, 
    :acot, :acoth,
    # exponentials
    :exp, :exp2, :exp10, :expm1, 
    :log, :log2, :log10, :log1p, 
    # roots
    :sqrt, :cbrt,
    # absolute values
    :abs2,
    # other
    :inv,
)
for op in ops_1_to_1_s
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_zero_global(::T) = false
end

# ops_1_to_1_f:
# ∂f/∂x   != 0
# ∂²f/∂x² == 0
ops_1_to_1_f = (
    :+, :-,
    :identity,
    :abs, :hypot,
    :deg2rad, :rad2deg,
    :mod2pi, :prevfloat, :nextfloat,
)
for op in ops_1_to_1_f
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_zero_global(::T) = true
end

# ops_1_to_1_z:
# ∂f/∂x   == 0
# ∂²f/∂x² == 0
ops_1_to_1_z = (
    :round, :floor, :ceil, :trunc,
    :sign,
)
for op in ops_1_to_1_z
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_zero_global(::T) = true
    SparseConnectivityTracer.is_seconder_zero_global(::T) = true
end

# Functions returning constant output
# that only depends on the input type.
# For the purpose of operator overloading,
# these are kept separate from ops_1_to_1_z.
ops_1_to_1_const = (
    :zero, :one,
    :eps, 
    :typemin, :typemax,
    :floatmin, :floatmax, :maxintfloat, 
)
for op in ops_1_to_1_const
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_zero_global(::T) = true
    SparseConnectivityTracer.is_seconder_zero_global(::T) = true
end

ops_1_to_1 = union(
    ops_1_to_1_s, 
    ops_1_to_1_f, 
    ops_1_to_1_z,
    ops_1_to_1_const,
)

##==================================#
# Operators for functions f: ℝ² → ℝ #
#===================================#

function is_firstder_arg1_zero_global end
function is_seconder_arg1_zero_global end
function is_firstder_arg2_zero_global end
function is_seconder_arg2_zero_global end
function is_crossder_zero_global end

# Fallbacks for local derivatives:
is_firstder_arg1_zero_local(f, x, y) = is_firstder_arg1_zero_global(f)
is_seconder_arg1_zero_local(f, x, y) = is_firstder_arg1_zero_global(f)
is_firstder_arg2_zero_local(f, x, y) = is_firstder_arg1_zero_global(f)
is_seconder_arg2_zero_local(f, x, y) = is_firstder_arg1_zero_global(f)
is_crossder_zero_local(f, x, y)      = is_firstder_arg1_zero_global(f)

# ops_2_to_1_ssc: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y != 0
ops_2_to_1_ssc = (
    :^, :hypot
)
for op in ops_2_to_1_ssc
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = false
    SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = false
    SparseConnectivityTracer.is_crossder_zero_global(::T)      = false
end

# ops_2_to_1_ssz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y == 0
ops_2_to_1_ssz = ()
# for op in ops_2_to_1_ssz
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = false
#     SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
#     SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = false
#     SparseConnectivityTracer.is_crossder_zero_global(::T)      = true
# end

# ops_2_to_1_sfc: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y != 0
ops_2_to_1_sfc = ()
# for op in ops_2_to_1_sfc
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = false
#     SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = true
#     SparseConnectivityTracer.is_crossder_zero_global(::T)      = false
# end

# ops_2_to_1_sfz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_sfz = ()
# for op in ops_2_to_1_sfz
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = false
#     SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = true
#     SparseConnectivityTracer.is_crossder_zero_global(::T)      = true
# end

# ops_2_to_1_fsc: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y != 0
ops_2_to_1_fsc = (
    :/, 
    # :ldexp,  # TODO: removed for now
)
for op in ops_2_to_1_fsc
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = true
    SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = false
    SparseConnectivityTracer.is_crossder_zero_global(::T)      = false
end

# gradient of x/y: [1/y -x/y²]
SparseConnectivityTracer.is_firstder_arg2_zero_local(::typeof(Base.:/), x, y) = iszero(x)

# ops_2_to_1_fsz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y == 0
ops_2_to_1_fsz = ()
# for op in ops_2_to_1_fsz
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = true
#     SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = false
#     SparseConnectivityTracer.is_crossder_zero_global(::T)      = true
# end

# ops_2_to_1_ffc: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y != 0
ops_2_to_1_ffc = (
    :*, 
)
for op in ops_2_to_1_ffc
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = true
    SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = true
    SparseConnectivityTracer.is_crossder_zero_global(::T)      = false
end

# gradient of x*y: [y x]
SparseConnectivityTracer.is_firstder_arg1_zero_local(::typeof(Base.:*), x, y) = iszero(y)
SparseConnectivityTracer.is_firstder_arg2_zero_local(::typeof(Base.:*), x, y) = iszero(x)

# ops_2_to_1_ffz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_ffz = (
    :+, :-,
    :mod, :rem,
    :min, :max,
)
for op in ops_2_to_1_ffz
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = true
    SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = true
    SparseConnectivityTracer.is_crossder_zero_global(::T)      = true
end

is_firstder_arg2_zero_local(::typeof(mod), x, y) = ifelse(y > 0, y > x, x > y)
is_firstder_arg2_zero_local(::typeof(rem), x, y) = ifelse(y > 0, y > x, x > y)

is_firstder_arg1_zero_local(::typeof(max), x, y) = x < y
is_firstder_arg2_zero_local(::typeof(max), x, y) = y < x

is_firstder_arg1_zero_local(::typeof(min), x, y) = x > y
is_firstder_arg2_zero_local(::typeof(min), x, y) = y > x

# ops_2_to_1_szz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    == 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_szz = ()
# for op in ops_2_to_1_szz
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = false
#     SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = true
#     SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = true
#     SparseConnectivityTracer.is_crossder_zero_global(::T)      = true
# end

# ops_2_to_1_zsz: 
# ∂f/∂x    == 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y == 0
ops_2_to_1_zsz = ()
# for op in ops_2_to_1_zsz
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = true
#     SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = true
#     SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = false
#     SparseConnectivityTracer.is_crossder_zero_global(::T)      = true
# end

# ops_2_to_1_fzz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    == 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_fzz = (
    :copysign, :flipsign,
)
for op in ops_2_to_1_fzz
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = true
    SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = true
    SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = true
    SparseConnectivityTracer.is_crossder_zero_global(::T)      = true
end

# ops_2_to_1_zfz: 
# ∂f/∂x    == 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_zfz = ()
# for op in ops_2_to_1_zfz
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = true
#     SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = true
#     SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = true
#     SparseConnectivityTracer.is_crossder_zero_global(::T)      = true
# end

# ops_2_to_1_zfz: 
# ∂f/∂x    == 0
# ∂²f/∂x²  == 0
# ∂f/∂y    == 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_zzz = (
    # division
    :div, :fld, :fld1, :cld, 
)
for op in ops_2_to_1_zzz
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_arg1_zero_global(::T) = true
    SparseConnectivityTracer.is_seconder_arg1_zero_global(::T) = true
    SparseConnectivityTracer.is_firstder_arg2_zero_global(::T) = true
    SparseConnectivityTracer.is_seconder_arg2_zero_global(::T) = true
    SparseConnectivityTracer.is_crossder_zero_global(::T)      = true
end

ops_2_to_1 = union(
    # Including second-order only
    ops_2_to_1_ssc,
    ops_2_to_1_ssz,

    # Including second- and first-order
    ops_2_to_1_sfc,
    ops_2_to_1_sfz,
    ops_2_to_1_fsc,
    ops_2_to_1_fsz,

    # Including first-order only
    ops_2_to_1_ffc,
    ops_2_to_1_ffz,

    # Including zero-order
    ops_2_to_1_szz,
    ops_2_to_1_zsz,
    ops_2_to_1_fzz,
    ops_2_to_1_zfz,
    ops_2_to_1_zzz,
)

##==================================#
# Operators for functions f: ℝ → ℝ² #
#===================================#

function is_firstder_out1_zero_global end
function is_seconder_out1_zero_global end
function is_firstder_out2_zero_global end
function is_seconder_out2_zero_global end

# Fallbacks for local derivatives:
is_seconder_out1_zero_local(f, x) = is_seconder_out1_zero_global(f)
is_firstder_out1_zero_local(f, x) = is_firstder_out1_zero_global(f)
is_firstder_out2_zero_local(f, x) = is_firstder_out2_zero_global(f)
is_seconder_out2_zero_local(f, x) = is_seconder_out2_zero_global(f)


# ops_1_to_2_ss: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² != 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² != 0
ops_1_to_2_ss = (
    :sincos,
    :sincosd,
    :sincospi,
)
for op in ops_1_to_2_ss
    T = typeof(eval(op))
    SparseConnectivityTracer.is_firstder_out1_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_out1_zero_global(::T) = false
    SparseConnectivityTracer.is_firstder_out2_zero_global(::T) = false
    SparseConnectivityTracer.is_seconder_out2_zero_global(::T) = false
end

# ops_1_to_2_sf: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² != 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² == 0
ops_1_to_2_sf = ()
# for op in ops_1_to_2_sf
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_out1_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out1_zero_global(::T) = false
#     SparseConnectivityTracer.is_firstder_out2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out2_zero_global(::T) = true
# end

# ops_1_to_2_sz: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² != 0
# ∂f₂/∂x   == 0
# ∂²f₂/∂x² == 0
ops_1_to_2_sz = ()
# for op in ops_1_to_2_sz
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_out1_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out1_zero_global(::T) = false
#     SparseConnectivityTracer.is_firstder_out2_zero_global(::T) = true
#     SparseConnectivityTracer.is_seconder_out2_zero_global(::T) = true
# end

# ops_1_to_2_fs: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² != 0
ops_1_to_2_fs = ()
# for op in ops_1_to_2_fs
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_out1_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out1_zero_global(::T) = true
#     SparseConnectivityTracer.is_firstder_out2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out2_zero_global(::T) = false
# end

# ops_1_to_2_ff: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² == 0
ops_1_to_2_ff = ()
# for op in ops_1_to_2_ff
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_out1_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out1_zero_global(::T) = true
#     SparseConnectivityTracer.is_firstder_out2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out2_zero_global(::T) = true
# end

# ops_1_to_2_fz: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   == 0
# ∂²f₂/∂x² == 0
ops_1_to_2_fz = (
    # :frexp,  # TODO: removed for now
)
# for op in ops_1_to_2_fz
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_out1_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out1_zero_global(::T) = true
#     SparseConnectivityTracer.is_firstder_out2_zero_global(::T) = true
#     SparseConnectivityTracer.is_seconder_out2_zero_global(::T) = true
# end

# ops_1_to_2_zs: 
# ∂f₁/∂x   == 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² != 0
ops_1_to_2_zs = ()
# for op in ops_1_to_2_zs
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_out1_zero_global(::T) = true
#     SparseConnectivityTracer.is_seconder_out1_zero_global(::T) = true
#     SparseConnectivityTracer.is_firstder_out2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out2_zero_global(::T) = false
# end

# ops_1_to_2_zf: 
# ∂f₁/∂x   == 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² == 0
ops_1_to_2_zf = ()
# for op in ops_1_to_2_zf
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_out1_zero_global(::T) = true
#     SparseConnectivityTracer.is_seconder_out1_zero_global(::T) = true
#     SparseConnectivityTracer.is_firstder_out2_zero_global(::T) = false
#     SparseConnectivityTracer.is_seconder_out2_zero_global(::T) = true
# end

# ops_1_to_2_zz: 
# ∂f₁/∂x   == 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   == 0
# ∂²f₂/∂x² == 0
ops_1_to_2_zz = ()
# for op in ops_1_to_2_zz
#     T = typeof(eval(op))
#     SparseConnectivityTracer.is_firstder_out1_zero_global(::T) = true
#     SparseConnectivityTracer.is_seconder_out1_zero_global(::T) = true
#     SparseConnectivityTracer.is_firstder_out2_zero_global(::T) = true
#     SparseConnectivityTracer.is_seconder_out2_zero_global(::T) = true
# end

ops_1_to_2 = union(
    ops_1_to_2_ss,
    ops_1_to_2_sf,
    ops_1_to_2_fs,
    ops_1_to_2_ff,
    ops_1_to_2_sz,
    ops_1_to_2_zs,
    ops_1_to_2_fz,
    ops_1_to_2_zf,
    ops_1_to_2_zz,   
)
#! format: on
