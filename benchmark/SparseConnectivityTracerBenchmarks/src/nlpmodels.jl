#=
Given an optimization problem `min f(x) s.t. c(x) <= 0`, we study

- the Jacobian of the constraints `c(x)`
- the Hessian of the Lagrangian `L(x,y) = f(x) + yᵀc(x)`

Package ecosystem overview: https://jso.dev/ecosystems/models/

- NLPModels.jl: abstract interface `AbstractNLPModel` for nonlinear optimization problems (with utilities to query objective, constraints, and their derivatives). See API at https://jso.dev/NLPModels.jl/stable/api/

- ADNLPModels.jl: concrete `ADNLPModel <: AbstractNLPModel` created from pure Julia code with autodiff
- NLPModelsJuMP.jl: concrete `MathOptNLPModel <: AbstractNLPModel` converted from a `JuMP.Model`

- OptimizationProblems.jl: suite of benchmark problems available in two formulations:
  - OptimizationProblems.ADNLPProblems: spits out `ADNLPModel`
  - OptimizationProblems.PureJuMP: spits out `JuMP.Model`
=#

# Obtained from `Symbol.(OptimizationProblems.meta[!, :name])`
const OPTIMIZATION_PROBLEM_NAMES = [
    :AMPGO02,
    :AMPGO03,
    :AMPGO04,
    :AMPGO05,
    :AMPGO06,
    :AMPGO07,
    :AMPGO08,
    :AMPGO09,
    :AMPGO10,
    :AMPGO11,
    :AMPGO12,
    :AMPGO13,
    :AMPGO14,
    :AMPGO15,
    :AMPGO18,
    :AMPGO20,
    :AMPGO21,
    :AMPGO22,
    :BOX2,
    :BOX3,
    :Dus2_1,
    :Dus2_3,
    :Dus2_9,
    :Duscube,
    :NZF1,
    :Shpak1,
    :Shpak2,
    :Shpak3,
    :Shpak4,
    :Shpak5,
    :Shpak6,
    :aircrfta,
    :allinit,
    :allinitc,
    :allinitu,
    :alsotame,
    :argauss,
    :arglina,
    :arglinb,
    :arglinc,
    :argtrig,
    :arwhead,
    :auglag,
    :avion2,
    :bard,
    :bdqrtic,
    :beale,
    :bearing,
    :bennett5,
    :biggs5,
    :biggs6,
    :booth,
    :boundary,
    :boxbod,
    :bqp1var,
    :britgas,
    :brownal,
    :brownbs,
    :brownden,
    :browngen1,
    :browngen2,
    :broyden3d,
    :broyden7d,
    :broydn7d,
    :brybnd,
    :bt1,
    :camshape,
    :catenary,
    # :catmix,
    :chain,
    :chainwoo,
    :channel,
    :chnrosnb_mod,
    :chwirut1,
    :chwirut2,
    :cliff,
    :clnlbeam,
    :clplatea,
    :clplateb,
    :clplatec,
    :controlinvestment,
    :cosine,
    :cragglvy,
    :cragglvy2,
    :curly,
    :curly10,
    :curly20,
    :curly30,
    :danwood,
    :dixmaane,
    :dixmaanf,
    :dixmaang,
    :dixmaanh,
    :dixmaani,
    :dixmaanj,
    :dixmaank,
    :dixmaanl,
    :dixmaanm,
    :dixmaann,
    :dixmaano,
    :dixmaanp,
    :dixon3dq,
    :dqdrtic,
    :dqrtic,
    :eckerle4,
    :edensch,
    :eg2,
    :elec,
    :engval1,
    :enso,
    :errinros_mod,
    :extrosnb,
    :fletcbv2,
    :fletcbv3_mod,
    :fletchcr,
    :fminsrf2,
    :freuroth,
    # :gasoil,
    :gauss1,
    :gauss2,
    :gauss3,
    :gaussian,
    :genbroydenb,
    :genbroydentri,
    :genhumps,
    :genrose,
    :genrose_nash,
    # :glider,
    :gulf,
    :hahn1,
    :helical,
    :hovercraft1d,
    :hs1,
    :hs10,
    :hs100,
    :hs101,
    :hs102,
    :hs103,
    :hs104,
    :hs105,
    :hs106,
    :hs107,
    :hs108,
    :hs109,
    :hs11,
    :hs110,
    :hs111,
    :hs112,
    :hs113,
    :hs114,
    :hs116,
    :hs117,
    :hs118,
    :hs119,
    :hs12,
    :hs13,
    :hs14,
    :hs15,
    :hs16,
    :hs17,
    :hs18,
    :hs19,
    :hs2,
    :hs20,
    :hs201,
    :hs21,
    :hs211,
    :hs219,
    :hs22,
    :hs220,
    :hs221,
    :hs222,
    :hs223,
    :hs224,
    :hs225,
    :hs226,
    :hs227,
    :hs228,
    :hs229,
    :hs23,
    :hs230,
    :hs231,
    :hs232,
    :hs233,
    :hs234,
    :hs235,
    :hs236,
    :hs237,
    :hs238,
    :hs239,
    :hs24,
    :hs240,
    :hs241,
    :hs242,
    :hs243,
    :hs244,
    :hs245,
    :hs246,
    :hs248,
    :hs249,
    :hs25,
    :hs250,
    :hs251,
    :hs252,
    :hs253,
    :hs254,
    :hs255,
    :hs256,
    :hs257,
    :hs258,
    :hs259,
    :hs26,
    :hs260,
    :hs261,
    :hs262,
    :hs263,
    :hs264,
    :hs265,
    :hs27,
    :hs28,
    :hs29,
    :hs3,
    :hs30,
    :hs31,
    :hs316,
    :hs317,
    :hs318,
    :hs319,
    :hs32,
    :hs320,
    :hs321,
    :hs322,
    :hs33,
    :hs34,
    :hs35,
    :hs36,
    :hs37,
    :hs378,
    :hs38,
    :hs39,
    :hs4,
    :hs40,
    :hs41,
    :hs42,
    :hs43,
    :hs44,
    :hs45,
    :hs46,
    :hs47,
    :hs48,
    :hs49,
    :hs5,
    :hs50,
    :hs51,
    :hs52,
    :hs53,
    :hs54,
    :hs55,
    :hs56,
    :hs57,
    :hs59,
    :hs6,
    :hs60,
    :hs61,
    :hs62,
    :hs63,
    :hs64,
    :hs65,
    :hs66,
    :hs68,
    :hs69,
    :hs7,
    :hs70,
    :hs71,
    :hs72,
    :hs73,
    :hs74,
    :hs75,
    :hs76,
    :hs77,
    :hs78,
    :hs79,
    :hs8,
    :hs80,
    :hs81,
    :hs83,
    :hs84,
    :hs86,
    :hs87,
    :hs9,
    :hs93,
    :hs95,
    :hs96,
    :hs97,
    :hs98,
    :hs99,
    :indef_mod,
    :integreq,
    :jennrichsampson,
    :kirby2,
    :kowosb,
    :lanczos1,
    :lanczos2,
    :lanczos3,
    :liarwhd,
    :lincon,
    :linsv,
    :marine,
    # :methanol,
    :meyer3,
    :mgh01feas,
    :mgh09,
    :mgh10,
    :mgh17,
    # :minsurf,
    :misra1a,
    :misra1b,
    :misra1c,
    :misra1d,
    :morebv,
    :nasty,
    :nazareth,
    :ncb20,
    :ncb20b,
    :nelson,
    :noncvxu2,
    :noncvxun,
    :nondia,
    :nondquar,
    :osborne1,
    :osborne2,
    :palmer1c,
    :palmer1d,
    :palmer2c,
    :palmer3c,
    :palmer4c,
    :palmer5c,
    :palmer5d,
    :palmer6c,
    :palmer7c,
    :palmer8c,
    :penalty1,
    :penalty2,
    :penalty3,
    # :pinene,
    :polygon,
    :polygon1,
    :polygon2,
    :polygon3,
    :powellbs,
    :powellsg,
    :power,
    :quartc,
    :rat42,
    :rat43,
    :robotarm,
    # :rocket,
    :rosenbrock,
    :rozman1,
    :sbrybnd,
    :schmvett,
    :scosine,
    :sinquad,
    :sparsine,
    :sparsqur,
    :spmsrtls,
    :srosenbr,
    # :steering,
    :structural,
    :tetra,
    :tetra_duct12,
    :tetra_duct15,
    :tetra_duct20,
    :tetra_foam5,
    :tetra_gear,
    :tetra_hook,
    :threepk,
    :thurber,
    :tointgss,
    # :torsion,
    :tquartic,
    :triangle,
    :triangle_deer,
    :triangle_pacman,
    :triangle_turtle,
    :tridia,
    :vardim,
    :variational,
    :vibrbeam,
    :watson,
    :woods,
    :zangwil3,
]

## SCT

#=
Here we use OptimizationProblems.ADNLPProblems because we need the problems in pure Julia.

https://jso.dev/OptimizationProblems.jl/stable/tutorial/#Problems-in-ADNLPModel-syntax:-ADNLPProblems
=#

function myconstraints(nlp::AbstractNLPModel, x::AbstractVector)
    c = similar(x, nlp.meta.ncon)
    NLPModels.cons!(nlp, x, c)
    return c
end

function mylagrangian(nlp::AbstractNLPModel, x::AbstractVector)
    f = NLPModels.obj(nlp, x)
    c = myconstraints(nlp, x)
    y = randn(length(c))
    L = f + dot(y, c)
    return L
end

function compute_jac_sparsity_sct(nlp::AbstractNLPModel)
    c = Base.Fix1(myconstraints, nlp)
    x0 = nlp.meta.x0
    jac_sparsity = ADTypes.jacobian_sparsity(c, x0, TracerSparsityDetector())
    return jac_sparsity
end

function compute_hess_sparsity_sct(nlp::AbstractNLPModel)
    L = Base.Fix1(mylagrangian, nlp)
    x0 = nlp.meta.x0
    hess_sparsity = ADTypes.hessian_sparsity(L, x0, TracerSparsityDetector())
    return hess_sparsity
end

function compute_jac_and_hess_sparsity_sct(name::Symbol)
    nlp = OptimizationProblems.ADNLPProblems.eval(name)()
    return compute_jac_sparsity_sct(nlp), compute_hess_sparsity_sct(nlp)
end

## Generic

function compute_jac_sparsity_and_value(nlp::AbstractNLPModel)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    x0 = nlp.meta.x0
    I, J = NLPModels.jac_structure(nlp)
    V = NLPModels.jac_coord(nlp, x0)
    jac_sparsity = sparse(I, J, ones(Bool, length(I)), m, n)
    jac = sparse(I, J, V, m, n)
    return jac_sparsity, jac
end

function compute_hess_sparsity_and_value(nlp::AbstractNLPModel)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    x0 = nlp.meta.x0
    yrand = rand(m)
    I, J = NLPModels.hess_structure(nlp)
    V = NLPModels.hess_coord(nlp, x0, yrand)
    hess_sparsity = sparse(Symmetric(sparse(I, J, ones(Bool, length(I)), n, n), :L))
    hess = sparse(Symmetric(sparse(I, J, V, n, n), :L))
    return hess_sparsity, hess
end

## JuMP

#=
Here we use OptimizationProblems.PureJuMP because JuMP is the ground truth, but we translate with NLPModelsJuMP to easily query the stuff we need. 

https://jso.dev/OptimizationProblems.jl/stable/tutorial/#Problems-in-JuMP-syntax:-PureJuMP
https://jso.dev/NLPModelsJuMP.jl/stable/tutorial/#MathOptNLPModel
=#

function compute_jac_and_hess_sparsity_and_value_jump(name::Symbol)
    nlp_jump = OptimizationProblems.PureJuMP.eval(name)()
    nlp = NLPModelsJuMP.MathOptNLPModel(nlp_jump)
    jac_sparsity, jac = compute_jac_sparsity_and_value(nlp)
    hess_sparsity, hess = compute_hess_sparsity_and_value(nlp)
    return ((jac_sparsity, jac), (hess_sparsity, hess))
end
