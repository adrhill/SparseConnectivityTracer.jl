## Operator definitions

# We use a system of letters to categorize operators:
# * z: first- and second-order derivatives (FOD, SOD) are zero
# * f: FOD is non-zero, SOD is zero
# * s: FOD is non-zero, SOD is non-zero

#! format: on

##=================================#
# Operators for functions f: ℝ → ℝ #
#==================================#

# ops_1_to_1_s: 
# ∂f/∂x   ≠ 0
# ∂²f/∂x² ≠ 0
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
# ∂f/∂x   ≠ 0
# ∂²f/∂x² = 0
ops_1_to_1_f = (
    # absolute values
    :abs
    # other
    :mod2pi
)

# ops_1_to_1_z:
# ∂f/∂x   = 0
# ∂²f/∂x² = 0
ops_1_to_1_z = (
    # rounding
    :round, :floor, :ceil, :trunc,4
    # other
    :sign
)

ops_1_to_1 = union(
    ops_1_to_1_s, 
    ops_1_to_1_f, 
    ops_1_to_1_z,
)

##==================================#
# Operators for functions f: ℝ² → ℝ #
#===================================#

ops_2_to_1 = (
    :+, :-, :*, :/, 
    # division
    :div, :fld, :cld, 
    # modulo
    :mod, :rem,
    # exponentials
    :ldexp, 
    # sign
    :copysign, :flipsign,
    # other
    :hypot,
)

# ops_2_to_1_sss: 
# ∂f/∂x    ≠ 0 
# ∂²f/∂x²  ≠ 0 
# ∂f/∂y    ≠ 0 
# ∂²f/∂y²  ≠ 0 
# ∂²f/∂x∂y ≠ 0 
ops_2_to_1_sss = ()

# ops_2_to_1_ssz: 
# ∂f/∂x    ≠ 0 
# ∂²f/∂x²  ≠ 0 
# ∂f/∂y    ≠ 0 
# ∂²f/∂y²  ≠ 0 
# ∂²f/∂x∂y = 0 
ops_2_to_1_ssz = ()

# ops_2_to_1_sfs: 
# ∂f/∂x    ≠ 0 
# ∂²f/∂x²  ≠ 0 
# ∂f/∂y    ≠ 0 
# ∂²f/∂y²  = 0 
# ∂²f/∂x∂y ≠ 0 
ops_2_to_1_sfs = ()

# ops_2_to_1_sfz: 
# ∂f/∂x    ≠ 0 
# ∂²f/∂x²  ≠ 0 
# ∂f/∂y    ≠ 0 
# ∂²f/∂y²  = 0 
# ∂²f/∂x∂y = 0 
ops_2_to_1_sfz = ()

# ops_2_to_1_szz: 
# ∂f/∂x    ≠ 0 
# ∂²f/∂x²  ≠ 0 
# ∂f/∂y    = 0 
# ∂²f/∂y²  = 0 
# ∂²f/∂x∂y ≠ 0 
ops_2_to_1_szz = ()

# ops_2_to_1_fss: 
# ∂f/∂x    ≠ 0 
# ∂²f/∂x²  = 0 
# ∂f/∂y    ≠ 0 
# ∂²f/∂y²  ≠ 0 
# ∂²f/∂x∂y ≠ 0 
ops_2_to_1_fss = ()

# ops_2_to_1_fsz: 
# ∂f/∂x    ≠ 0 
# ∂²f/∂x²  = 0 
# ∂f/∂y    ≠ 0 
# ∂²f/∂y²  ≠ 0 
# ∂²f/∂x∂y = 0 
ops_2_to_1_fsz = ()

# ops_2_to_1_zsz: 
# ∂f/∂x    = 0 
# ∂²f/∂x²  = 0 
# ∂f/∂y    ≠ 0 
# ∂²f/∂y²  ≠ 0 
# ∂²f/∂x∂y = 0 
ops_2_to_1_fss = ()

# ops_2_to_1_fzz: 
# ∂f/∂x    ≠ 0 
# ∂²f/∂x²  = 0 
# ∂f/∂y    = 0 
# ∂²f/∂y²  = 0 
# ∂²f/∂x∂y = 0 
ops_2_to_1_fzz = ()

ops_2_to_1_ss = (
    # trigonometric
    :sincos,
    :sincosd,
    :sincospi,
    # exponentials
    :frexp,
)

ops_2_to_1 = union(
    ops_2_to_1_ss,
    ops_2_to_1_sf,
    ops_2_to_1_sz,
    ops_2_to_1_fs,
    ops_2_to_1_ff,
    ops_2_to_1_fz,
    ops_2_to_1_zs,
    ops_2_to_1_zf,
    ops_2_to_1_zz,   
)


##==================================#
# Operators for functions f: ℝ → ℝ² #
#===================================#

# ops_1_to_2_ss: 
# ∂f₁/∂x   ≠ 0 
# ∂²f₁/∂x² ≠ 0 
# ∂f₂/∂x   ≠ 0 
# ∂²f₂/∂x² ≠ 0 
ops_1_to_2_ss = (
    # trigonometric
    :sincos,
    :sincosd,
    :sincospi,
    # exponentials
    :frexp,
)

# ops_1_to_2_sf: 
# ∂f₁/∂x   ≠ 0 
# ∂²f₁/∂x² ≠ 0 
# ∂f₂/∂x   ≠ 0 
# ∂²f₂/∂x² = 0 
ops_1_to_2_sf = ()

# ops_1_to_2_sz: 
# ∂f₁/∂x   ≠ 0 
# ∂²f₁/∂x² ≠ 0 
# ∂f₂/∂x   = 0 
# ∂²f₂/∂x² = 0 
ops_1_to_2_sz = ()

# ops_1_to_2_fs: 
# ∂f₁/∂x   ≠ 0 
# ∂²f₁/∂x² = 0 
# ∂f₂/∂x   ≠ 0 
# ∂²f₂/∂x² ≠ 0 
ops_1_to_2_fs = ()

# ops_1_to_2_ff: 
# ∂f₁/∂x   ≠ 0 
# ∂²f₁/∂x² = 0 
# ∂f₂/∂x   ≠ 0 
# ∂²f₂/∂x² = 0 
ops_1_to_2_ff = ()

# ops_1_to_2_fz: 
# ∂f₁/∂x   ≠ 0 
# ∂²f₁/∂x² = 0 
# ∂f₂/∂x   = 0 
# ∂²f₂/∂x² = 0 
ops_1_to_2_fz = ()

# ops_1_to_2_zs: 
# ∂f₁/∂x   = 0 
# ∂²f₁/∂x² = 0 
# ∂f₂/∂x   ≠ 0 
# ∂²f₂/∂x² ≠ 0 
ops_1_to_2_zs = ()

# ops_1_to_2_zf: 
# ∂f₁/∂x   = 0 
# ∂²f₁/∂x² = 0 
# ∂f₂/∂x   ≠ 0 
# ∂²f₂/∂x² = 0 
ops_1_to_2_zf = ()

# ops_1_to_2_zz: 
# ∂f₁/∂x   = 0 
# ∂²f₁/∂x² = 0 
# ∂f₂/∂x   = 0 
# ∂²f₂/∂x² = 0 
ops_1_to_2_zz = ()

ops_1_to_2 = union(
    ops_1_to_2_ss,
    ops_1_to_2_sf,
    ops_1_to_2_sz,
    ops_1_to_2_fs,
    ops_1_to_2_ff,
    ops_1_to_2_fz,
    ops_1_to_2_zs,
    ops_1_to_2_zf,
    ops_1_to_2_zz,   
)
#! format: on
