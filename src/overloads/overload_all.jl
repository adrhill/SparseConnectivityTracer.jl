# for overload in (
#     overload_gradient_1_to_1,
#     overload_gradient_2_to_1,
#     overload_gradient_1_to_2,
#     overload_hessian_1_to_1,
#     overload_hessian_2_to_1,
#     overload_hessian_1_to_2,
# )
#     @eval function $overload(ops::Tuple)
#         for op in ops
#             $overload(op)
#         end
#     end
# end

# overload_gradient_1_to_1(ops_1_to_1)
# overload_gradient_2_to_1(ops_2_to_1)
# overload_gradient_1_to_2(ops_1_to_2)
# overload_hessian_1_to_1(ops_1_to_1)
# overload_hessian_2_to_1(ops_2_to_1)
# overload_hessian_1_to_2(ops_1_to_2)

for op in ops_1_to_1
    overload_gradient_1_to_1(op)
    overload_hessian_1_to_1(op)
end
for op in ops_2_to_1
    overload_gradient_2_to_1(op)
    overload_hessian_2_to_1(op)
end
for op in ops_1_to_2
    overload_gradient_1_to_2(op)
    overload_hessian_1_to_2(op)
end
