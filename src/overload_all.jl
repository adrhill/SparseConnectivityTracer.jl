function overload_all(M)
    exprs_1_to_1 = [
        quote
            $(overload_connectivity_1_to_1(M, op))
            $(overload_gradient_1_to_1(M, op))
            $(overload_hessian_1_to_1(M, op))
        end for op in nameof.(list_operators_1_to_1(Val(M)))
    ]
    exprs_1_to_1_dual = [
        quote
            $(overload_connectivity_1_to_1_dual(M, op))
            $(overload_gradient_1_to_1_dual(M, op))
            $(overload_hessian_1_to_1_dual(M, op))
        end for op in nameof.(list_operators_1_to_1(Val(M)))
    ]
    exprs_2_to_1 = [
        quote
            $(overload_connectivity_2_to_1(M, op))
            $(overload_gradient_2_to_1(M, op))
            $(overload_hessian_2_to_1(M, op))
        end for op in nameof.(list_operators_2_to_1(Val(M)))
    ]
    exprs_2_to_1_dual = [
        quote
            $(overload_connectivity_2_to_1_dual(M, op))
            $(overload_gradient_2_to_1_dual(M, op))
            $(overload_hessian_2_to_1_dual(M, op))
        end for op in nameof.(list_operators_2_to_1(Val(M)))
    ]
    exprs_1_to_2 = [
        quote
            $(overload_connectivity_1_to_2(M, op))
            $(overload_gradient_1_to_2(M, op))
            $(overload_hessian_1_to_2(M, op))
        end for op in nameof.(list_operators_1_to_2(Val(M)))
    ]
    exprs_1_to_2_dual = [
        quote
            $(overload_connectivity_1_to_2_dual(M, op))
            $(overload_gradient_1_to_2_dual(M, op))
            $(overload_hessian_1_to_2_dual(M, op))
        end for op in nameof.(list_operators_1_to_2(Val(M)))
    ]
    return quote
        $(exprs_1_to_1...)
        $(exprs_1_to_1_dual...)
        $(exprs_2_to_1...)
        $(exprs_2_to_1_dual...)
        $(exprs_1_to_2...)
        $(exprs_1_to_2_dual...)
    end
end

eval(overload_all(:Base))
