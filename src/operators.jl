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

# ops_1_to_1_s: 
# ∂f/∂x   != 0
# ∂²f/∂x² != 0
ops_1_to_1_s = (
    # trigonometric functions
    :deg2rad, :rad2deg,
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
    :inv, :hypot,
)

# ops_1_to_1_f:
# ∂f/∂x   != 0
# ∂²f/∂x² == 0
ops_1_to_1_f = (
    :+, :-,
    # absolute values
    :abs,
    # other
    :mod2pi, :prevfloat, :nextfloat,
)

# ops_1_to_1_z:
# ∂f/∂x   == 0
# ∂²f/∂x² == 0
ops_1_to_1_z = (
    # rounding
    :round, :floor, :ceil, :trunc,
    # other
    :sign,
)

# Functions returning constant output
# that only depends on the input type.
# For the purpose of operator overloading,
# these are kept separate from ops_1_to_1_z.
ops_1_to_1_const = (
    :zero, :one,
    :eps, :floatmin, :floatmax, :maxintfloat, :typemax
)

ops_1_to_1 = union(
    ops_1_to_1_s, 
    ops_1_to_1_f, 
    ops_1_to_1_z,
    ops_1_to_1_const,
)

##==================================#
# Operators for functions f: ℝ² → ℝ #
#===================================#

# ops_2_to_1_ssc: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y != 0
ops_2_to_1_ssc = (
    :hypot,
)

# ops_2_to_1_ssz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y == 0
ops_2_to_1_ssz = ()

# ops_2_to_1_sfc: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y != 0
ops_2_to_1_sfc = ()

# ops_2_to_1_sfz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_sfz = ()

# ops_2_to_1_fsc: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y != 0
ops_2_to_1_fsc = (
    :/, 
    # exponentials
    :ldexp, 
)

# ops_2_to_1_fsz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y == 0
ops_2_to_1_fsz = ()

# ops_2_to_1_ffc: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y != 0
ops_2_to_1_ffc = (
    :*, 
)

# ops_2_to_1_ffz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_ffz = (
    :+, :-, 
)

# ops_2_to_1_szz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    == 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_szz = ()

# ops_2_to_1_zsz: 
# ∂f/∂x    == 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y == 0
ops_2_to_1_zsz = ()

# ops_2_to_1_fzz: 
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    == 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_fzz = ()

# ops_2_to_1_zfz: 
# ∂f/∂x    == 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_zfz = ()

# ops_2_to_1_zfz: 
# ∂f/∂x    == 0
# ∂²f/∂x²  == 0
# ∂f/∂y    == 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_zzz = (
    # division
    :div, :fld, :fld1, :cld, 
    # modulo and friends
    :mod, :rem,
    # sign
    :copysign, :flipsign,
)

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

# ops_1_to_2_ss: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² != 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² != 0
ops_1_to_2_ss = (
    # trigonometric
    :sincos,
    :sincosd,
    :sincospi,
)

# ops_1_to_2_sf: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² != 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² == 0
ops_1_to_2_sf = ()

# ops_1_to_2_sz: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² != 0
# ∂f₂/∂x   == 0
# ∂²f₂/∂x² == 0
ops_1_to_2_sz = ()

# ops_1_to_2_fs: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² != 0
ops_1_to_2_fs = ()

# ops_1_to_2_ff: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² == 0
ops_1_to_2_ff = ()

# ops_1_to_2_fz: 
# ∂f₁/∂x   != 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   == 0
# ∂²f₂/∂x² == 0
ops_1_to_2_fz = (
    :frexp,
)

# ops_1_to_2_zs: 
# ∂f₁/∂x   == 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² != 0
ops_1_to_2_zs = ()

# ops_1_to_2_zf: 
# ∂f₁/∂x   == 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   != 0
# ∂²f₂/∂x² == 0
ops_1_to_2_zf = ()

# ops_1_to_2_zz: 
# ∂f₁/∂x   == 0
# ∂²f₁/∂x² == 0
# ∂f₂/∂x   == 0
# ∂²f₂/∂x² == 0
ops_1_to_2_zz = ()

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
