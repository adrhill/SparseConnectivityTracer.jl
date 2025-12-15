module SparseConnectivityTracerBenchmarks

module ODE
    include("brusselator.jl")
    export Brusselator!, brusselator_2d_loop!
end

end # module SparseConnectivityTracerBenchmarks
