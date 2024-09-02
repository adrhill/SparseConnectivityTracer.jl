SCT = SparseConnectivityTracer

## 1-to-1

@noinline function gradient_tracer_1_to_1(
    t::T, is_der1_zero::Bool
) where {T<:GradientTracer}
    if is_der1_zero && !isemptytracer(t)
        return myempty(T)
    else
        return t
    end
end

function gradient_tracer_1_to_1_inner(
    p::P, is_der1_zero::Bool
) where {P<:IndexSetGradientPattern}
    return P(gradient_tracer_1_to_1_inner(gradient(p), is_der1_zero)) # return pattern
end

# This is only required because it is called by HessianTracer with IndexSetHessianPattern
# Otherwise, we would just have the method on IndexSetGradientPattern above.
function gradient_tracer_1_to_1_inner(
    s::S, is_der1_zero::Bool
) where {S<:AbstractSet{<:Integer}}
    if is_der1_zero
        return myempty(S)
    else
        return s # return set
    end
end

function generate_code_gradient_1_to_1(M::Symbol, f)
    fname = nameof(f)
    is_der1_zero_g = is_der1_zero_global(f)

    expr_gradienttracer = quote
        function $M.$fname(t::$SCT.GradientTracer)
            return $SCT.gradient_tracer_1_to_1(t, $is_der1_zero_g)
        end
    end

    expr_dual = if is_der1_zero_g
        quote
            function $M.$fname(d::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(d)
                return $M.$fname(x)
            end
        end
    else
        quote
            function $M.$fname(d::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(d)
                p_out = $M.$fname(x)

                t = $SCT.tracer(d)
                is_der1_zero = $SCT.is_der1_zero_local($M.$fname, x)
                t_out = $SCT.gradient_tracer_1_to_1(t, is_der1_zero)
                return $SCT.Dual(p_out, t_out)
            end
        end
    end
    return Expr(:block, expr_gradienttracer, expr_dual)
end

## 2-to-1

@noinline function gradient_tracer_2_to_1(
    tx::T, ty::T, is_der1_arg1_zero::Bool, is_der1_arg2_zero::Bool
) where {T<:GradientTracer}
    # TODO: add tests for isempty
    if tx.isempty && ty.isempty
        return tx # empty tracer
    elseif ty.isempty
        return gradient_tracer_1_to_1(tx, is_der1_arg1_zero)
    elseif tx.isempty
        return gradient_tracer_1_to_1(ty, is_der1_arg2_zero)
    else
        g_out = gradient_tracer_2_to_1_inner(
            pattern(tx), pattern(ty), is_der1_arg1_zero, is_der1_arg2_zero
        )
        return T(g_out) # return tracer
    end
end

function gradient_tracer_2_to_1_inner(
    px::P, py::P, is_der1_arg1_zero::Bool, is_der1_arg2_zero::Bool
) where {P<:IndexSetGradientPattern}
    return P(
        gradient_tracer_2_to_1_inner(
            gradient(px), gradient(py), is_der1_arg1_zero, is_der1_arg2_zero
        ),
    ) # return pattern
end

# This is only required because it is called by HessianTracer with IndexSetHessianPattern
# Otherwise, we would just have the method on IndexSetGradientPattern above.
function gradient_tracer_2_to_1_inner(
    sx::S, sy::S, is_der1_arg1_zero::Bool, is_der1_arg2_zero::Bool
) where {S<:AbstractSet{<:Integer}}
    if is_der1_arg1_zero && is_der1_arg2_zero
        return myempty(S)
    elseif !is_der1_arg1_zero && is_der1_arg2_zero
        return sx
    elseif is_der1_arg1_zero && !is_der1_arg2_zero
        return sy
    else
        return union(sx, sy) # return set
    end
end

function generate_code_gradient_2_to_1(M::Symbol, f)
    fname = nameof(f)
    is_der1_arg1_zero_g = is_der1_arg1_zero_global(f)
    is_der1_arg2_zero_g = is_der1_arg2_zero_global(f)

    ## GradientTracer
    expr_gradienttracer = quote
        function $M.$fname(tx::T, ty::T) where {T<:$SCT.GradientTracer}
            return $SCT.gradient_tracer_2_to_1(
                tx, ty, $is_der1_arg1_zero_g, $is_der1_arg2_zero_g
            )
        end

        function $M.$fname(tx::$SCT.GradientTracer, ::Real)
            return $SCT.gradient_tracer_1_to_1(tx, $is_der1_arg1_zero_g)
        end

        function $M.$fname(::Real, ty::$SCT.GradientTracer)
            return $SCT.gradient_tracer_1_to_1(ty, $is_der1_arg2_zero_g)
        end
    end

    ## Dual
    expr_dual_dual = if is_der1_arg1_zero_g && is_der1_arg2_zero_g
        quote
            function $M.$fname(dx::D, dy::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(dx)
                y = $SCT.primal(dy)
                return $M.$fname(x, y)
            end
        end
    else
        quote
            function $M.$fname(dx::D, dy::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(dx)
                y = $SCT.primal(dy)
                p_out = $M.$fname(x, y)

                tx = $SCT.tracer(dx)
                ty = $SCT.tracer(dy)
                is_der1_arg1_zero = $SCT.is_der1_arg1_zero_local($M.$fname, x, y)
                is_der1_arg2_zero = $SCT.is_der1_arg2_zero_local($M.$fname, x, y)
                t_out = $SCT.gradient_tracer_2_to_1(
                    tx, ty, is_der1_arg1_zero, is_der1_arg2_zero
                )
                return $SCT.Dual(p_out, t_out)
            end
        end
    end
    expr_dual_real = if is_der1_arg1_zero_g
        quote
            function $M.$fname(
                dx::D, y::Real
            ) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(dx)
                return $M.$fname(x, y)
            end
        end
    else
        quote
            function $M.$fname(
                dx::D, y::Real
            ) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(dx)
                p_out = $M.$fname(x, y)

                tx = $SCT.tracer(dx)
                is_der1_arg1_zero = $SCT.is_der1_arg1_zero_local($M.$fname, x, y)
                t_out = $SCT.gradient_tracer_1_to_1(tx, is_der1_arg1_zero)
                return $SCT.Dual(p_out, t_out)
            end
        end
    end
    expr_real_dual = if is_der1_arg2_zero_g
        quote
            function $M.$fname(
                x::Real, dy::D
            ) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                y = $SCT.primal(dy)
                return $M.$fname(x, y)
            end
        end
    else
        quote
            function $M.$fname(
                x::Real, dy::D
            ) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                y = $SCT.primal(dy)
                p_out = $M.$fname(x, y)

                ty = $SCT.tracer(dy)
                is_der1_arg2_zero = $SCT.is_der1_arg2_zero_local($M.$fname, x, y)
                t_out = $SCT.gradient_tracer_1_to_1(ty, is_der1_arg2_zero)
                return $SCT.Dual(p_out, t_out)
            end
        end
    end

    return Expr(:block, expr_gradienttracer, expr_dual_dual, expr_dual_real, expr_real_dual)
end

## 1-to-2

@noinline function gradient_tracer_1_to_2(
    t::T, is_der1_out1_zero::Bool, is_der1_out2_zero::Bool
) where {T<:GradientTracer}
    if isemptytracer(t) # TODO: add test
        return (t, t)
    else
        t_out1 = gradient_tracer_1_to_1(t, is_der1_out1_zero)
        t_out2 = gradient_tracer_1_to_1(t, is_der1_out2_zero)
        return (t_out1, t_out2)
    end
end

function generate_code_gradient_1_to_2(M::Symbol, f)
    fname = nameof(f)
    is_der1_out1_zero_g = is_der1_out1_zero_global(f)
    is_der1_out2_zero_g = is_der1_out2_zero_global(f)

    expr_gradienttracer = quote
        function $M.$fname(t::$SCT.GradientTracer)
            return $SCT.gradient_tracer_1_to_2(t, $is_der1_out1_zero_g, $is_der1_out2_zero_g)
        end
    end

    expr_dual = if is_der1_out1_zero_g && is_der1_out2_zero_g
        quote
            function $M.$fname(d::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(d)
                return $M.$fname(x)
            end
        end
    else
        quote
            function $M.$fname(d::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
                x = $SCT.primal(d)
                p_out1, p_out2 = $M.$fname(x)

                t = $SCT.tracer(d)
                is_der1_out2_zero = $SCT.is_der1_out2_zero_local($M.$fname, x)
                is_der1_out1_zero = $SCT.is_der1_out1_zero_local($M.$fname, x)
                t_out1, t_out2 = $SCT.gradient_tracer_1_to_2(
                    t, is_der1_out1_zero, is_der1_out2_zero
                )
                return ($SCT.Dual(p_out1, t_out1), $SCT.Dual(p_out2, t_out2))  # TODO: this was wrong, add test
            end
        end
    end

    return Expr(:block, expr_gradienttracer, expr_dual)
end
