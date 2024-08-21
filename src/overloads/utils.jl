for overload in (
    :overload_gradient_1_to_1,
    :overload_gradient_2_to_1,
    :overload_gradient_1_to_2,
    :overload_hessian_1_to_1,
    :overload_hessian_2_to_1,
    :overload_hessian_1_to_2,
)
    @eval function $overload(M::Symbol, ops::Union{AbstractVector,Tuple})
        exprs = [$overload(M, op) for op in ops]
        return Expr(:block, exprs...)
    end
end

## Overload operators
eval(overload_gradient_1_to_1(:Base, ops_1_to_1))
eval(overload_gradient_2_to_1(:Base, ops_2_to_1))
eval(overload_gradient_1_to_2(:Base, ops_1_to_2))
eval(overload_hessian_1_to_1(:Base, ops_1_to_1))
eval(overload_hessian_2_to_1(:Base, ops_2_to_1))
eval(overload_hessian_1_to_2(:Base, ops_1_to_2))

## List operators for later testing
test_operators_1_to_1(::Val{:Base}) = ops_1_to_1
test_operators_2_to_1(::Val{:Base}) = ops_2_to_1
test_operators_1_to_2(::Val{:Base}) = ops_1_to_2
