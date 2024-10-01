module SparseConnectivityTracerForwardDiffExt

if isdefined(Base, :get_extension)
    import SparseConnectivityTracer as SCT
    using ForwardDiff: ForwardDiff
else
    import ..SparseConnectivityTracer as SCT
    using ..ForwardDiff: ForwardDiff
end

# Overload 2-to-1 functions on ForwardDiff.Dual
eval(SCT.generate_code_2_to_1(:Base, SCT.ops_2_to_1, ForwardDiff.Dual))

end # module
