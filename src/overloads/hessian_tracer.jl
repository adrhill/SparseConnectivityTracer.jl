## 1-to-1

function hessian_tracer_1_to_1(
    t::T, is_firstder_zero::Bool, is_secondder_zero::Bool
) where {T<:HessianTracer}
    if isemptytracer(t) # TODO: add test
        return t
    else
        g_out, h_out = hessian_tracer_1_to_1_inner(
            gradient(t), hessian(t), is_firstder_zero, is_secondder_zero
        )
        return T(g_out, h_out) # return tracer
    end
end

function hessian_tracer_1_to_1_inner(
    sg::G, sh::H, is_firstder_zero::Bool, is_secondder_zero::Bool
) where {I<:Integer,G<:AbstractSet{I},H<:AbstractSet{Tuple{I,I}}}
    sg_out = gradient_tracer_1_to_1_inner(sg, is_firstder_zero)
    sh_out = if is_firstder_zero && is_secondder_zero
        myempty(H)
    elseif !is_firstder_zero && is_secondder_zero
        sh
    elseif is_firstder_zero && !is_secondder_zero
        union_product!(myempty(H), sg, sg)
    else
        union_product!(copy(sh), sg, sg)
    end
    return sg_out, sh_out # return sets
end

function overload_hessian_1_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::$SCT.HessianTracer)
            is_firstder_zero = $SCT.is_firstder_zero_global($M.$op)
            is_seconder_zero = $SCT.is_seconder_zero_global($M.$op)
            return @noinline $SCT.hessian_tracer_1_to_1(
                t, is_firstder_zero, is_seconder_zero
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

            t = $SCT.tracer(d)
            is_firstder_zero = $SCT.is_firstder_zero_local($M.$op, x)
            is_seconder_zero = $SCT.is_seconder_zero_local($M.$op, x)
            t_out = @noinline $SCT.hessian_tracer_1_to_1(
                t, is_firstder_zero, is_seconder_zero
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
        g_out, h_out = hessian_tracer_2_to_1_inner(
            gradient(tx),
            hessian(tx),
            gradient(ty),
            hessian(ty),
            is_firstder_arg1_zero,
            is_secondder_arg1_zero,
            is_firstder_arg2_zero,
            is_secondder_arg2_zero,
            is_crossder_zero,
        )
        return T(g_out, h_out) # return tracer
    end
end

function hessian_tracer_2_to_1_inner(
    sgx::G,
    shx::H,
    sgy::G,
    shy::H,
    is_firstder_arg1_zero::Bool,
    is_secondder_arg1_zero::Bool,
    is_firstder_arg2_zero::Bool,
    is_secondder_arg2_zero::Bool,
    is_crossder_zero::Bool,
) where {I<:Integer,G<:AbstractSet{I},H<:AbstractSet{Tuple{I,I}}}
    sg_out = gradient_tracer_2_to_1_inner(
        sgx, sgy, is_firstder_arg1_zero, is_firstder_arg2_zero
    )
    sh_out = myempty(H)
    !is_firstder_arg1_zero && union!(sh_out, shx)  # hessian alpha
    !is_firstder_arg2_zero && union!(sh_out, shy)  # hessian beta
    !is_secondder_arg1_zero && union_product!(sh_out, sgx, sgx)  # product alpha
    !is_secondder_arg2_zero && union_product!(sh_out, sgy, sgy)  # product beta
    !is_crossder_zero && union_product!(sh_out, sgx, sgy)  # cross product 1
    !is_crossder_zero && union_product!(sh_out, sgy, sgx)  # cross product 2
    return sg_out, sh_out # return sets
end

function overload_hessian_2_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(tx::T, ty::T) where {T<:$SCT.HessianTracer}
            is_firstder_arg1_zero = $SCT.is_firstder_arg1_zero_global($M.$op)
            is_seconder_arg1_zero = $SCT.is_seconder_arg1_zero_global($M.$op)
            is_firstder_arg2_zero = $SCT.is_firstder_arg2_zero_global($M.$op)
            is_seconder_arg2_zero = $SCT.is_seconder_arg2_zero_global($M.$op)
            is_crossder_zero = $SCT.is_crossder_zero_global($M.$op)
            return @noinline $SCT.hessian_tracer_2_to_1(
                tx,
                ty,
                is_firstder_arg1_zero,
                is_seconder_arg1_zero,
                is_firstder_arg2_zero,
                is_seconder_arg2_zero,
                is_crossder_zero,
            )
        end

        function $M.$op(tx::$SCT.HessianTracer, y::Real)
            is_firstder_arg1_zero = $SCT.is_firstder_arg1_zero_global($M.$op)
            is_seconder_arg1_zero = $SCT.is_seconder_arg1_zero_global($M.$op)
            return @noinline $SCT.hessian_tracer_1_to_1(
                tx, is_firstder_arg1_zero, is_seconder_arg1_zero
            )
        end

        function $M.$op(x::Real, ty::$SCT.HessianTracer)
            is_firstder_arg2_zero = $SCT.is_firstder_arg2_zero_global($M.$op)
            is_seconder_arg2_zero = $SCT.is_seconder_arg2_zero_global($M.$op)
            return @noinline $SCT.hessian_tracer_1_to_1(
                ty, is_firstder_arg2_zero, is_seconder_arg2_zero
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
            t_out = @noinline $SCT.hessian_tracer_2_to_1(
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

            tx = $SCT.tracer(dx)
            is_firstder_arg1_zero = $SCT.is_firstder_arg1_zero_local($M.$op, x, y)
            is_seconder_arg1_zero = $SCT.is_seconder_arg1_zero_local($M.$op, x, y)
            t_out = @noinline $SCT.hessian_tracer_1_to_1(
                tx, is_firstder_arg1_zero, is_seconder_arg1_zero
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(x::Real, dy::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)

            tx = $SCT.tracer(dy)
            is_firstder_arg2_zero = $SCT.is_firstder_arg2_zero_local($M.$op, x, y)
            is_seconder_arg2_zero = $SCT.is_seconder_arg2_zero_local($M.$op, x, y)
            t_out = @noinline $SCT.hessian_tracer_1_to_1(
                tx, is_firstder_arg2_zero, is_seconder_arg2_zero
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
    if isemptytracer(t) # TODO: add test
        return (t, t)
    else
        t_out1 = hessian_tracer_1_to_1(t, is_firstder_out1_zero, is_seconder_out1_zero)
        t_out2 = hessian_tracer_1_to_1(t, is_firstder_out2_zero, is_seconder_out2_zero)
        return (t_out1, t_out2)
    end
end

function overload_hessian_1_to_2(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::$SCT.HessianTracer)
            is_firstder_out1_zero = $SCT.is_firstder_out1_zero_global($M.$op)
            is_seconder_out1_zero = $SCT.is_seconder_out1_zero_global($M.$op)
            is_firstder_out2_zero = $SCT.is_firstder_out2_zero_global($M.$op)
            is_seconder_out2_zero = $SCT.is_seconder_out2_zero_global($M.$op)
            return @noinline $SCT.hessian_tracer_1_to_2(
                t,
                is_firstder_out1_zero,
                is_seconder_out1_zero,
                is_firstder_out2_zero,
                is_seconder_out2_zero,
            )
        end
    end
end

function overload_hessian_1_to_2_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(d::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(d)
            p_out1, p_out2 = $M.$op(x)

            is_firstder_out1_zero = $SCT.is_firstder_out1_zero_local($M.$op, x)
            is_seconder_out1_zero = $SCT.is_seconder_out1_zero_local($M.$op, x)
            is_firstder_out2_zero = $SCT.is_firstder_out2_zero_local($M.$op, x)
            is_seconder_out2_zero = $SCT.is_seconder_out2_zero_local($M.$op, x)
            t_out1, t_out2 = @noinline $SCT.hessian_tracer_1_to_2(
                d,
                is_firstder_out1_zero,
                is_seconder_out1_zero,
                is_firstder_out2_zero,
                is_seconder_out2_zero,
            )
            return ($SCT.Dual(p_out1, t_out1), $SCT.Dual(p_out2, t_out2))
        end
    end
end

## Special cases

## Exponent (requires extra types)
for S in (Integer, Rational, Irrational{:ℯ})
    Base.:^(t::T, ::S) where {T<:HessianTracer} = hessian_tracer_1_to_1(t, false, false)
    Base.:^(::S, t::T) where {T<:HessianTracer} = hessian_tracer_1_to_1(t, false, false)

    function Base.:^(d::D, y::S) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(d)
        t = hessian_tracer_1_to_1(tracer(d), false, false)
        return Dual(x^y, t)
    end
    function Base.:^(x::S, d::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        y = primal(d)
        t = hessian_tracer_1_to_1(tracer(d), false, false)
        return Dual(x^y, t)
    end
end

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:HessianTracer} = myempty(T)

## Random numbers
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:HessianTracer} = myempty(T)  # TODO: was missing Base, add tests
