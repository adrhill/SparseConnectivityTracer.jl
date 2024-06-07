## 1-to-1

function hessian_tracer_1_to_1(
    t::T, is_firstder_zero::Bool, is_secondder_zero::Bool
) where {T<:HessianTracer}
    if t.isempty # TODO: add test
        return t
    else
        pattern = hessian_tracer_1_to_1_pattern(
            t.pattern, is_firstder_zero, is_secondder_zero
        )
        return T(pattern) # return tracer
    end
end

function hessian_tracer_1_to_1_pattern(
    p::P, is_firstder_zero::Bool, is_secondder_zero::Bool
) where {P<:SimpleVectorAndMatrixIndexSetPattern}
    set_grad, set_hessian = hessian_tracer_1_to_1_set(
        gradient(p), hessian(p), is_firstder_zero, is_secondder_zero
    )
    return P(set_grad, set_hessian) # return pattern
end

function hessian_tracer_1_to_1_set(
    sg::SG, sh::SH, is_firstder_zero::Bool, is_secondder_zero::Bool
) where {SG,SH}
    sg_out = gradient_tracer_1_to_1_set(sg, is_firstder_zero)
    sh_out = if is_firstder_zero && is_secondder_zero
        myempty(SH)
    elseif !is_firstder_zero && is_secondder_zero
        sh
    elseif is_firstder_zero && !is_secondder_zero
        union_product!(myempty(SH), sg, sg)
    else
        union_product!(copy(sh), sg, sg)
    end
    return (sg_out, sh_out) # return sets
end

function overload_hessian_1_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::$SCT.HessianTracer)
            return $SCT.hessian_tracer_1_to_1(
                t,
                $SCT.is_firstder_zero_global($M.$op),
                $SCT.is_seconder_zero_global($M.$op),
            )
        end
    end
end

function overload_hessian_1_to_1_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(d::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(d)
            p_out = $M.$op(x)
            t_out = $SCT.hessian_tracer_1_to_1(
                $SCT.tracer(d),
                $SCT.is_firstder_zero_local($M.$op, x),
                $SCT.is_seconder_zero_local($M.$op, x),
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 2-to-1

function hessian_tracer_2_to_1(
    tx::T,
    ty::T,
    is_firstder_arg1_zero::Bool,
    is_secondder_arg1_zero::Bool,
    is_firstder_arg2_zero::Bool,
    is_secondder_arg2_zero::Bool,
    is_crossder_zero::Bool,
) where {T<:HessianTracer}
    # TODO: add tests for isempty
    if tx.isempty && ty.isempty
        return tx # empty tracer
    elseif ty.isempty
        return hessian_tracer_1_to_1(tx, is_firstder_arg1_zero, is_secondder_arg1_zero)
    elseif tx.isempty
        return hessian_tracer_1_to_1(ty, is_firstder_arg2_zero, is_secondder_arg2_zero)
    else
        pattern = hessian_tracer_2_to_1_pattern(
            tx.pattern,
            ty.pattern,
            is_firstder_arg1_zero,
            is_secondder_arg1_zero,
            is_firstder_arg2_zero,
            is_secondder_arg2_zero,
            is_crossder_zero,
        )
        return T(pattern) # return tracer
    end
end

function hessian_tracer_2_to_1_pattern(
    px::P,
    py::P,
    is_firstder_arg1_zero::Bool,
    is_secondder_arg1_zero::Bool,
    is_firstder_arg2_zero::Bool,
    is_secondder_arg2_zero::Bool,
    is_crossder_zero::Bool,
) where {P<:SimpleVectorAndMatrixIndexSetPattern}
    set_grad, set_hessian = hessian_tracer_2_to_1_set(
        gradient(px),
        hessian(px),
        gradient(py),
        hessian(py),
        is_firstder_arg1_zero,
        is_secondder_arg1_zero,
        is_firstder_arg2_zero,
        is_secondder_arg2_zero,
        is_crossder_zero,
    )
    return P(set_grad, set_hessian) # return pattern
end

function hessian_tracer_2_to_1_set(
    sgx::SG,
    shx::SH,
    sgy::SG,
    shy::SH,
    is_firstder_arg1_zero::Bool,
    is_secondder_arg1_zero::Bool,
    is_firstder_arg2_zero::Bool,
    is_secondder_arg2_zero::Bool,
    is_crossder_zero::Bool,
) where {SG,SH}
    sg_out = gradient_tracer_2_to_1_set(
        sgx, sgy, is_firstder_arg1_zero, is_firstder_arg2_zero
    )
    sh_out = myempty(SH)
    !is_firstder_arg1_zero && union!(sh_out, shx)  # hessian alpha
    !is_firstder_arg2_zero && union!(sh_out, shy)  # hessian beta
    !is_secondder_arg1_zero && union_product!(sh_out, sgx, sgx)  # product alpha
    !is_secondder_arg2_zero && union_product!(sh_out, sgy, sgy)  # product beta
    !is_crossder_zero && union_product!(sh_out, sgx, sgy)  # cross product 1
    !is_crossder_zero && union_product!(sh_out, sgy, sgx)  # cross product 2
    return (sg_out, sh_out) # return sets
end

function overload_hessian_2_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(tx::T, ty::T) where {T<:$SCT.HessianTracer}
            return $SCT.hessian_tracer_2_to_1(
                tx,
                ty,
                $SCT.is_firstder_arg1_zero_global($M.$op),
                $SCT.is_seconder_arg1_zero_global($M.$op),
                $SCT.is_firstder_arg2_zero_global($M.$op),
                $SCT.is_seconder_arg2_zero_global($M.$op),
                $SCT.is_crossder_zero_global($M.$op),
            )
        end

        function $M.$op(tx::$SCT.HessianTracer, y::Real)
            return $SCT.hessian_tracer_1_to_1(
                tx,
                $SCT.is_firstder_arg1_zero_global($M.$op),
                $SCT.is_seconder_arg1_zero_global($M.$op),
            )
        end

        function $M.$op(x::Real, ty::$SCT.HessianTracer)
            return $SCT.hessian_tracer_1_to_1(
                ty,
                $SCT.is_firstder_arg2_zero_global($M.$op),
                $SCT.is_seconder_arg2_zero_global($M.$op),
            )
        end
    end
end

function overload_hessian_2_to_1_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(dx::D, dy::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(dx)
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)
            t_out = $SCT.hessian_tracer_2_to_1(
                $SCT.tracer(dx),
                $SCT.tracer(dy),
                $SCT.is_firstder_arg1_zero_local($M.$op, x, y),
                $SCT.is_seconder_arg1_zero_local($M.$op, x, y),
                $SCT.is_firstder_arg2_zero_local($M.$op, x, y),
                $SCT.is_seconder_arg2_zero_local($M.$op, x, y),
                $SCT.is_crossder_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(dx::D, y::Real) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(dx)
            p_out = $M.$op(x, y)
            t_out = $SCT.hessian_tracer_1_to_1(
                $SCT.tracer(dx),
                $SCT.is_firstder_arg1_zero_local($M.$op, x, y),
                $SCT.is_seconder_arg1_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(x::Real, dy::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)
            t_out = $SCT.hessian_tracer_1_to_1(
                $SCT.tracer(dy),
                $SCT.is_firstder_arg2_zero_local($M.$op, x, y),
                $SCT.is_seconder_arg2_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 1-to-2

function hessian_tracer_1_to_2(
    t::T,
    is_firstder_out1_zero::Bool,
    is_seconder_out1_zero::Bool,
    is_firstder_out2_zero::Bool,
    is_seconder_out2_zero::Bool,
) where {T<:HessianTracer}
    if t.isempty # TODO: add test
        return (t, t)
    else
        pattern1, pattern2 = hessian_tracer_1_to_2_pattern(
            t.pattern,
            is_firstder_out1_zero,
            is_seconder_out1_zero,
            is_firstder_out2_zero,
            is_seconder_out2_zero,
        )
        return (T(pattern1), T(pattern2)) # return tracers
    end
end

function hessian_tracer_1_to_2_pattern(
    p::P,
    is_firstder_out1_zero::Bool,
    is_seconder_out1_zero::Bool,
    is_firstder_out2_zero::Bool,
    is_seconder_out2_zero::Bool,
) where {P<:SimpleVectorAndMatrixIndexSetPattern}
    (set_grad1, set_hessian1), (set_grad2, set_hessian2) = hessian_tracer_1_to_2_set(
        gradient(p),
        hessian(p),
        is_firstder_out1_zero,
        is_seconder_out1_zero,
        is_firstder_out2_zero,
        is_seconder_out2_zero,
    )
    pattern1 = P(set_grad1, set_hessian1)
    pattern2 = P(set_grad2, set_hessian2)
    return (pattern1, pattern2) # return patterns
end

function hessian_tracer_1_to_2_set(
    sg::SG,
    sh::SH,
    is_firstder_out1_zero::Bool,
    is_secondder_out1_zero::Bool,
    is_firstder_out2_zero::Bool,
    is_secondder_out2_zero::Bool,
) where {SG,SH}
    sg_out1, sh_out1 = hessian_tracer_1_to_1_set(
        sg, sh, is_firstder_out1_zero, is_secondder_out1_zero
    )
    sg_out2, sh_out2 = hessian_tracer_1_to_1_set(
        sg, sh, is_firstder_out2_zero, is_secondder_out2_zero
    )
    return ((sg_out1, sh_out1), (sg_out2, sh_out2)) # return sets
end

function overload_hessian_1_to_2(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::$SCT.HessianTracer)
            return $SCT.hessian_tracer_1_to_2(
                t,
                $SCT.is_firstder_out1_zero_global($M.$op),
                $SCT.is_seconder_out1_zero_global($M.$op),
                $SCT.is_firstder_out2_zero_global($M.$op),
                $SCT.is_seconder_out2_zero_global($M.$op),
            )
        end
    end
end

function overload_hessian_1_to_2_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(d::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(d)
            p1_out, p2_out = $M.$op(x)
            t1_out, t2_out = $SCT.hessian_tracer_1_to_2(
                d,
                $SCT.is_firstder_out1_zero_local($M.$op, x),
                $SCT.is_seconder_out1_zero_local($M.$op, x),
                $SCT.is_firstder_out2_zero_local($M.$op, x),
                $SCT.is_seconder_out2_zero_local($M.$op, x),
            )
            return ($SCT.Dual(p1_out, t1_out), $SCT.Dual(p2_out, t2_out))
        end
    end
end

## Special cases

## Exponent (requires extra types)
for S in (Integer, Rational, Irrational{:ℯ})
    function Base.:^(t::T, ::S) where {T<:HessianTracer}
        pattern = hessian_tracer_1_to_1_pattern(t.pattern, false, false)
        return T(pattern)
    end
    function Base.:^(::S, t::T) where {T<:HessianTracer}
        pattern = hessian_tracer_1_to_1_pattern(t.pattern, false, false)
        return T(pattern)
    end

    function Base.:^(d::D, y::S) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(d)
        t = tracer(d)
        pattern = hessian_tracer_1_to_1_pattern(t.pattern, false, false)
        return Dual(x^y, T(pattern))
    end
    function Base.:^(x::S, d::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        y = primal(d)
        t = tracer(d)
        pattern = hessian_tracer_1_to_1_pattern(t.pattern, false, false)
        return Dual(x^y, T(pattern))
    end
end

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:HessianTracer} = myempty(T)

## Random numbers
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:HessianTracer} = myempty(T)  # TODO: was missing Base, add tests
