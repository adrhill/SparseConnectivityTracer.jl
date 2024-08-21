SCT = SparseConnectivityTracer

## 1-to-1

# 𝟙[∇γ]  = 𝟙[∂φ]⋅𝟙[∇α]
# 𝟙[∇²γ] = 𝟙[∂φ]⋅𝟙[∇²α] ∨ 𝟙[∂²φ]⋅(𝟙[∇α] ∨ 𝟙[∇α]ᵀ)

@noinline function hessian_tracer_1_to_1(
    t::T, is_der1_zero::Bool, is_der2_zero::Bool
) where {P<:AbstractHessianPattern,T<:HessianTracer{P}}
    if isemptytracer(t) # TODO: add test
        return t
    else
        p_out = hessian_tracer_1_to_1_inner(
            pattern(t), is_der1_zero, is_der2_zero, shared(P)
        )
        return T(p_out) # return tracer
    end
end

function hessian_tracer_1_to_1_inner(
    p::P, is_der1_zero::Bool, is_der2_zero::Bool, ::NotShared
) where {P<:AbstractHessianPattern}
    g = gradient(p)
    h = hessian(p)

    g_out = gradient_tracer_1_to_1_inner(g, is_der1_zero) # 𝟙[∇γ] = 𝟙[∂φ]⋅𝟙[∇α]
    h_out = if is_der1_zero && is_der2_zero # 𝟙[∇²γ] = 0
        myempty(h)
    elseif !is_der1_zero && is_der2_zero # 𝟙[∇²γ] = 𝟙[∂φ]⋅𝟙[∇²α]
        h
    elseif is_der1_zero && !is_der2_zero # 𝟙[∇²γ] = 𝟙[∇α] ∨ 𝟙[∇α]ᵀ
        # TODO: this branch of the code currently isn't tested.
        # Covering it would require a scalar 1-to-1 function with local overloads, 
        # such that ∂f/∂x == 0 and ∂²f/∂x² != 0.
        union_product!(myempty(h), g, g)
    else # !is_der1_zero && !is_der2_zero,  𝟙[∇²γ] = 𝟙[∇²α] ∨ (𝟙[∇α] ∨ 𝟙[∇α]ᵀ)
        union_product!(copy(h), g, g)
    end
    return P(g_out, h_out) # return pattern
end

# NOTE: mutates argument p and should arguably be called `hessian_tracer_1_to_1_inner!`
function hessian_tracer_1_to_1_inner(
    p::P, is_der1_zero::Bool, is_der2_zero::Bool, ::Shared
) where {P<:AbstractHessianPattern}
    g = gradient(p)
    g_out = gradient_tracer_1_to_1_inner(g, is_der1_zero)

    # shared Hessian patterns can't remove second-order information, only add to it.
    h = hessian(p)
    h_out = if is_der2_zero  # 𝟙[∇²γ] = 𝟙[∂φ]⋅𝟙[∇²α]
        h
    else # 𝟙[∇²γ] = 𝟙[∇²α] ∨ (𝟙[∇α] ∨ 𝟙[∇α]ᵀ)
        union_product!(h, g, g)
    end
    return P(g_out, h_out) # return pattern
end

function overload_hessian_1_to_1(M::Symbol, f)
    fname = nameof(f)
    is_der1_zero_g = is_der1_zero_global(f)
    is_der2_zero_g = is_der2_zero_global(f)

    expr_hessiantracer = quote
        ## HessianTracer
        function $M.$fname(t::$SCT.HessianTracer)
            return $SCT.hessian_tracer_1_to_1(t, $is_der1_zero_g, $is_der2_zero_g)
        end
    end

    expr_dual = if is_der1_zero_g && is_der1_zero_g
        quote
            function $M.$fname(d::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(d)
                return $M.$fname(x)
            end
        end
    else
        quote
            function $M.$fname(d::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(d)
                p_out = $M.$fname(x)

                t = $SCT.tracer(d)
                is_der1_zero = $SCT.is_der1_zero_local($M.$fname, x)
                is_der2_zero = $SCT.is_der2_zero_local($M.$fname, x)
                t_out = $SCT.hessian_tracer_1_to_1(t, is_der1_zero, is_der2_zero)
                return $SCT.Dual(p_out, t_out)
            end
        end
    end

    return Expr(:block, expr_hessiantracer, expr_dual)
end

## 2-to-1

@noinline function hessian_tracer_2_to_1(
    tx::T,
    ty::T,
    is_der1_arg1_zero::Bool,
    is_der2_arg1_zero::Bool,
    is_der1_arg2_zero::Bool,
    is_der2_arg2_zero::Bool,
    is_der_cross_zero::Bool,
) where {P<:AbstractHessianPattern,T<:HessianTracer{P}}
    # TODO: add tests for isempty
    if tx.isempty && ty.isempty
        return tx # empty tracer
    elseif ty.isempty
        return hessian_tracer_1_to_1(tx, is_der1_arg1_zero, is_der2_arg1_zero)
    elseif tx.isempty
        return hessian_tracer_1_to_1(ty, is_der1_arg2_zero, is_der2_arg2_zero)
    else
        p_out = hessian_tracer_2_to_1_inner(
            pattern(tx),
            pattern(ty),
            is_der1_arg1_zero,
            is_der2_arg1_zero,
            is_der1_arg2_zero,
            is_der2_arg2_zero,
            is_der_cross_zero,
            shared(P),
        )
        return T(p_out) # return tracer
    end
end

function hessian_tracer_2_to_1_inner(
    px::P,
    py::P,
    is_der1_arg1_zero::Bool,
    is_der2_arg1_zero::Bool,
    is_der1_arg2_zero::Bool,
    is_der2_arg2_zero::Bool,
    is_der_cross_zero::Bool,
    ::NotShared,
) where {P<:AbstractHessianPattern}
    gx, hx = gradient(px), hessian(px)
    gy, hy = gradient(py), hessian(py)
    g_out = gradient_tracer_2_to_1_inner(gx, gy, is_der1_arg1_zero, is_der1_arg2_zero)
    h_out = myempty(hx)
    !is_der1_arg1_zero && myunion!(h_out, hx)  # hessian alpha
    !is_der1_arg2_zero && myunion!(h_out, hy)  # hessian beta
    !is_der2_arg1_zero && union_product!(h_out, gx, gx)  # product alpha
    !is_der2_arg2_zero && union_product!(h_out, gy, gy)  # product beta
    !is_der_cross_zero && union_product!(h_out, gx, gy)  # cross product 1
    !is_der_cross_zero && union_product!(h_out, gy, gx)  # cross product 2
    return P(g_out, h_out) # return pattern
end

# NOTE: mutates arguments px and py and should arguably be called `hessian_tracer_1_to_1_inner!`
function hessian_tracer_2_to_1_inner(
    px::P,
    py::P,
    is_der1_arg1_zero::Bool,
    is_der2_arg1_zero::Bool,
    is_der1_arg2_zero::Bool,
    is_der2_arg2_zero::Bool,
    is_der_cross_zero::Bool,
    ::Shared,
) where {P<:AbstractHessianPattern}
    gx, hx = gradient(px), hessian(px)
    gy, hy = gradient(py), hessian(py)

    hx !== hy && error("Expected shared Hessians, got $hx, $hy.")
    h_out = hx # union of hx and hy can be skipped since they are the same object
    g_out = gradient_tracer_2_to_1_inner(gx, gy, is_der1_arg1_zero, is_der1_arg2_zero)

    !is_der2_arg1_zero && union_product!(h_out, gx, gx)  # product alpha
    !is_der2_arg2_zero && union_product!(h_out, gy, gy)  # product beta
    !is_der_cross_zero && union_product!(h_out, gx, gy)  # cross product 1
    !is_der_cross_zero && union_product!(h_out, gy, gx)  # cross product 2
    return P(g_out, h_out) # return pattern
end

function overload_hessian_2_to_1(M::Symbol, f)
    fname = nameof(f)
    is_der1_arg1_zero_g = is_der1_arg1_zero_global(f)
    is_der2_arg1_zero_g = is_der2_arg1_zero_global(f)
    is_der1_arg2_zero_g = is_der1_arg2_zero_global(f)
    is_der2_arg2_zero_g = is_der2_arg2_zero_global(f)
    is_der_cross_zero_g = is_der_cross_zero_global(f)

    ## HessianTracer
    expr_hessiantracer = quote
        function $M.$fname(tx::T, ty::T) where {T<:$SCT.HessianTracer}
            return $SCT.hessian_tracer_2_to_1(
                tx,
                ty,
                $is_der1_arg1_zero_g,
                $is_der2_arg1_zero_g,
                $is_der1_arg2_zero_g,
                $is_der2_arg2_zero_g,
                $is_der_cross_zero_g,
            )
        end

        function $M.$fname(tx::$SCT.HessianTracer, y::Real)
            return $SCT.hessian_tracer_1_to_1(tx, $is_der1_arg1_zero_g, $is_der2_arg1_zero_g)
        end

        function $M.$fname(x::Real, ty::$SCT.HessianTracer)
            return $SCT.hessian_tracer_1_to_1(ty, $is_der1_arg2_zero_g, $is_der2_arg2_zero_g)
        end
    end

    ## Dual
    expr_dual_dual =
        if is_der1_arg1_zero_g &&
            is_der2_arg1_zero_g &&
            is_der1_arg2_zero_g &&
            is_der2_arg2_zero_g &&
            is_der_cross_zero_g
            quote
                function $M.$fname(
                    dx::D, dy::D
                ) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                    x = $SCT.primal(dx)
                    y = $SCT.primal(dy)
                    return $M.$fname(x, y)
                end
            end
        else
            quote
                function $M.$fname(
                    dx::D, dy::D
                ) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                    x = $SCT.primal(dx)
                    y = $SCT.primal(dy)
                    p_out = $M.$fname(x, y)

                    tx = $SCT.tracer(dx)
                    ty = $SCT.tracer(dy)
                    is_der1_arg1_zero = $SCT.is_der1_arg1_zero_local($M.$fname, x, y)
                    is_der2_arg1_zero = $SCT.is_der2_arg1_zero_local($M.$fname, x, y)
                    is_der1_arg2_zero = $SCT.is_der1_arg2_zero_local($M.$fname, x, y)
                    is_der2_arg2_zero = $SCT.is_der2_arg2_zero_local($M.$fname, x, y)
                    is_der_cross_zero = $SCT.is_der_cross_zero_local($M.$fname, x, y)
                    t_out = $SCT.hessian_tracer_2_to_1(
                        tx,
                        ty,
                        is_der1_arg1_zero,
                        is_der2_arg1_zero,
                        is_der1_arg2_zero,
                        is_der2_arg2_zero,
                        is_der_cross_zero,
                    )
                    return $SCT.Dual(p_out, t_out)
                end
            end
        end
    expr_dual_real = if is_der1_arg1_zero_g && is_der2_arg1_zero_g
        quote
            function $M.$fname(dx::D, y::Real) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(dx)
                return $M.$fname(x, y)
            end
        end
    else
        quote
            function $M.$fname(dx::D, y::Real) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(dx)
                p_out = $M.$fname(x, y)

                tx = $SCT.tracer(dx)
                is_der1_arg1_zero = $SCT.is_der1_arg1_zero_local($M.$fname, x, y)
                is_der2_arg1_zero = $SCT.is_der2_arg1_zero_local($M.$fname, x, y)
                t_out = $SCT.hessian_tracer_1_to_1(tx, is_der1_arg1_zero, is_der2_arg1_zero)
                return $SCT.Dual(p_out, t_out)
            end
        end
    end
    expr_real_dual = if is_der1_arg2_zero_g && is_der2_arg2_zero_g
        quote
            function $M.$fname(x::Real, dy::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                y = $SCT.primal(dy)
                return $M.$fname(x, y)
            end
        end
    else
        quote
            function $M.$fname(x::Real, dy::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                y = $SCT.primal(dy)
                p_out = $M.$fname(x, y)

                ty = $SCT.tracer(dy)
                is_der1_arg2_zero = $SCT.is_der1_arg2_zero_local($M.$fname, x, y)
                is_der2_arg2_zero = $SCT.is_der2_arg2_zero_local($M.$fname, x, y)
                t_out = $SCT.hessian_tracer_1_to_1(ty, is_der1_arg2_zero, is_der2_arg2_zero)
                return $SCT.Dual(p_out, t_out)
            end
        end
    end

    return Expr(:block, expr_hessiantracer, expr_dual_dual, expr_dual_real, expr_real_dual)
end

## 1-to-2

@noinline function hessian_tracer_1_to_2(
    t::T,
    is_der1_out1_zero::Bool,
    is_der2_out1_zero::Bool,
    is_der1_out2_zero::Bool,
    is_der2_out2_zero::Bool,
) where {T<:HessianTracer}
    if isemptytracer(t) # TODO: add test
        return (t, t)
    else
        t_out1 = hessian_tracer_1_to_1(t, is_der1_out1_zero, is_der2_out1_zero)
        t_out2 = hessian_tracer_1_to_1(t, is_der1_out2_zero, is_der2_out2_zero)
        return (t_out1, t_out2)
    end
end

function overload_hessian_1_to_2(M::Symbol, f)
    fname = nameof(f)
    is_der1_out1_zero_g = is_der1_out1_zero_global(f)
    is_der2_out1_zero_g = is_der2_out1_zero_global(f)
    is_der1_out2_zero_g = is_der1_out2_zero_global(f)
    is_der2_out2_zero_g = is_der2_out2_zero_global(f)

    expr_hessiantracer = quote
        function $M.$fname(t::$SCT.HessianTracer)
            return $SCT.hessian_tracer_1_to_2(
                t,
                $is_der1_out1_zero_g,
                $is_der2_out1_zero_g,
                $is_der1_out2_zero_g,
                $is_der2_out2_zero_g,
            )
        end
    end

    expr_dual =
        if is_der1_out1_zero_g &&
            is_der2_out1_zero_g &&
            is_der1_out2_zero_g &&
            is_der2_out2_zero_g
            quote
                function $M.$fname(d::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                    x = $SCT.primal(d)
                    return $M.$fname(x)
                end
            end
        else
            quote
                function $M.$fname(d::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
                    x = $SCT.primal(d)
                    p_out1, p_out2 = $M.$fname(x)

                    is_der1_out1_zero = $SCT.is_der1_out1_zero_local($M.$fname, x)
                    is_der2_out1_zero = $SCT.is_der2_out1_zero_local($M.$fname, x)
                    is_der1_out2_zero = $SCT.is_der1_out2_zero_local($M.$fname, x)
                    is_der2_out2_zero = $SCT.is_der2_out2_zero_local($M.$fname, x)
                    t_out1, t_out2 = $SCT.hessian_tracer_1_to_2(
                        d,
                        is_der1_out1_zero,
                        is_der2_out1_zero,
                        is_der1_out2_zero,
                        is_der2_out2_zero,
                    )
                    return ($SCT.Dual(p_out1, t_out1), $SCT.Dual(p_out2, t_out2))
                end
            end
        end

    return Expr(:block, expr_hessiantracer, expr_dual)
end
