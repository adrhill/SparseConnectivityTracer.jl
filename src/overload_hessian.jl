## 1-to-1

function hessian_tracer_1_to_1(
    t::T, is_firstder_zero::Bool, is_secondder_zero::Bool
) where {T<:HessianTracer}
    sg, sh = gradient(t), hessian(t)
    sg_out, sh_out = hessian_tracer_1_to_1(sg, sh, is_firstder_zero, is_secondder_zero)
    return T(sg_out, sh_out)
end

function hessian_tracer_1_to_1(
    sg::SG, sh::SH, is_firstder_zero::Bool, is_secondder_zero::Bool
) where {I,SG<:AbstractSet{<:I},SH<:AbstractSet{<:Tuple{I,I}}}
    sg_out = gradient_tracer_1_to_1(sg, is_firstder_zero)
    sh_out = if is_firstder_zero && is_secondder_zero
        myempty(SH)
    elseif !is_firstder_zero && is_secondder_zero
        sh
    elseif is_firstder_zero && !is_secondder_zero
        union_product!(myempty(SH), sg, sg)
    else
        union_product!(copy(sh), sg, sg)
    end
    return (sg_out, sh_out)
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
    sgx, shx = gradient(tx), hessian(tx)
    sgy, shy = gradient(ty), hessian(ty)
    sg_out, sh_out = hessian_tracer_2_to_1(
        sgx,
        shx,
        sgy,
        shy,
        is_firstder_arg1_zero,
        is_secondder_arg1_zero,
        is_firstder_arg2_zero,
        is_secondder_arg2_zero,
        is_crossder_zero,
    )
    return T(sg_out, sh_out)
end

function hessian_tracer_2_to_1(
    sgx::SG,
    shx::SH,
    sgy::SG,
    shy::SH,
    is_firstder_arg1_zero::Bool,
    is_secondder_arg1_zero::Bool,
    is_firstder_arg2_zero::Bool,
    is_secondder_arg2_zero::Bool,
    is_crossder_zero::Bool,
) where {I,SG<:AbstractSet{I},SH<:AbstractSet{<:Tuple{I,I}}}
    sg_out = gradient_tracer_2_to_1(sgx, sgy, is_firstder_arg1_zero, is_firstder_arg2_zero)
    sh_out = myempty(SH)
    !is_firstder_arg1_zero && union!(sh_out, shx)  # hessian alpha
    !is_firstder_arg2_zero && union!(sh_out, shy)  # hessian beta
    !is_secondder_arg1_zero && union_product!(sh_out, sgx, sgx)  # product alpha
    !is_secondder_arg2_zero && union_product!(sh_out, sgy, sgy)  # product beta
    !is_crossder_zero && union_product!(sh_out, sgx, sgy)  # cross product 1
    !is_crossder_zero && union_product!(sh_out, sgy, sgx)  # cross product 2
    return (sg_out, sh_out)
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
    sg, sh = gradient(t), hessian(t)
    (sg_out1, sh_out1), (sg_out2, sh_out2) = hessian_tracer_1_to_2(
        sg,
        sh,
        is_firstder_out1_zero,
        is_seconder_out1_zero,
        is_firstder_out2_zero,
        is_seconder_out2_zero,
    )
    return (T(sg_out1, sh_out1), T(sg_out2, sh_out2))
end

function hessian_tracer_1_to_2(
    sg::SG,
    sh::SH,
    is_firstder_out1_zero::Bool,
    is_secondder_out1_zero::Bool,
    is_firstder_out2_zero::Bool,
    is_secondder_out2_zero::Bool,
) where {I,SG<:AbstractSet{I},SH<:AbstractSet{<:Tuple{I,I}}}
    sg_out1, sh_out1 = hessian_tracer_1_to_1(
        sg, sh, is_firstder_out1_zero, is_secondder_out1_zero
    )
    sg_out2, sh_out2 = hessian_tracer_1_to_1(
        sg, sh, is_firstder_out2_zero, is_secondder_out2_zero
    )
    return ((sg_out1, sh_out1), (sg_out2, sh_out2))
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
    function Base.:^(tx::T, y::S) where {T<:HessianTracer}
        return T(
            gradient(tx), union_product!(copy(hessian(tx)), gradient(tx), gradient(tx))
        )
    end
    function Base.:^(x::S, ty::T) where {T<:HessianTracer}
        return T(
            gradient(ty), union_product!(copy(hessian(ty)), gradient(ty), gradient(ty))
        )
    end

    function Base.:^(dx::D, y::S) where {P,T<:HessianTracer,D<:Dual{P,T}}
        return Dual(
            primal(dx)^y,
            T(gradient(dx), union_product!(copy(hessian(dx)), gradient(dx), gradient(dx))),
        )
    end
    function Base.:^(x::S, dy::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        return Dual(
            x^primal(dy),
            T(gradient(dy), union_product!(copy(hessian(dy)), gradient(dy), gradient(dy))),
        )
    end
end

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:HessianTracer} = myempty(T)

## Random numbers
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:HessianTracer} = myempty(T)  # TODO: was missing Base, add tests
