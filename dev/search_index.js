var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API Reference","title":"API Reference","text":"CurrentModule = Main\nCollapsedDocStrings = true","category":"page"},{"location":"api/#API-Reference","page":"API Reference","title":"API Reference","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"","category":"page"},{"location":"api/#Interface","page":"API Reference","title":"Interface","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"pattern\nTracerSparsityDetector","category":"page"},{"location":"api/#SparseConnectivityTracer.pattern","page":"API Reference","title":"SparseConnectivityTracer.pattern","text":"pattern(f, JacobianTracer, x)\n\nComputes the sparsity pattern of the Jacobian of y = f(x).\n\npattern(f, ConnectivityTracer, x)\n\nEnumerates inputs x and primal outputs y = f(x) and returns sparse matrix C of size (m, n) where C[i, j] is true if the compute graph connects the i-th entry in y to the j-th entry in x.\n\nExample\n\njulia> x = rand(3);\n\njulia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];\n\njulia> pattern(f, ConnectivityTracer, x)\n3×3 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 4 stored entries:\n 1  ⋅  ⋅\n 1  1  ⋅\n ⋅  ⋅  1\n\n\n\n\n\npattern(f!, y, JacobianTracer, x)\n\nComputes the sparsity pattern of the Jacobian of f!(y, x).\n\npattern(f!, y, ConnectivityTracer, x)\n\nEnumerates inputs x and primal outputs y after f!(y, x) and returns sparse matrix C of size (m, n) where C[i, j] is true if the compute graph connects the i-th entry in y to the j-th entry in x.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseConnectivityTracer.TracerSparsityDetector","page":"API Reference","title":"SparseConnectivityTracer.TracerSparsityDetector","text":"TracerSparsityDetector <: ADTypes.AbstractSparsityDetector\n\nSingleton struct for integration with the sparsity detection framework of ADTypes.jl.\n\nExample\n\njulia> using ADTypes, SparseConnectivityTracer\n\njulia> ADTypes.jacobian_sparsity(diff, rand(4), TracerSparsityDetector())\n3×4 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 6 stored entries:\n 1  1  ⋅  ⋅\n ⋅  1  1  ⋅\n ⋅  ⋅  1  1\n\n\n\n\n\n","category":"type"},{"location":"api/#Internals","page":"API Reference","title":"Internals","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"SparseConnectivityTracer works by pushing Number types called tracers through generic functions. Currently, two tracer types are provided:","category":"page"},{"location":"api/","page":"API Reference","title":"API Reference","text":"JacobianTracer\nConnectivityTracer","category":"page"},{"location":"api/#SparseConnectivityTracer.JacobianTracer","page":"API Reference","title":"SparseConnectivityTracer.JacobianTracer","text":"JacobianTracer(indexset) <: Number\n\nNumber type keeping track of input indices of previous computations with non-zero derivatives.\n\nSee also the convenience constructor tracer. For a higher-level interface, refer to pattern.\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseConnectivityTracer.ConnectivityTracer","page":"API Reference","title":"SparseConnectivityTracer.ConnectivityTracer","text":"ConnectivityTracer(indexset) <: Number\n\nNumber type keeping track of input indices of previous computations.\n\nSee also the convenience constructor tracer. For a higher-level interface, refer to pattern.\n\n\n\n\n\n","category":"type"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Utilities to create tracers:","category":"page"},{"location":"api/","page":"API Reference","title":"API Reference","text":"tracer\ntrace_input","category":"page"},{"location":"api/#SparseConnectivityTracer.tracer","page":"API Reference","title":"SparseConnectivityTracer.tracer","text":"tracer(JacobianTracer, index)\ntracer(JacobianTracer, indices)\ntracer(ConnectivityTracer, index)\ntracer(ConnectivityTracer, indices)\n\nConvenience constructor for JacobianTracer ConnectivityTracer from input indices.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseConnectivityTracer.trace_input","page":"API Reference","title":"SparseConnectivityTracer.trace_input","text":"trace_input(JacobianTracer, x)\ntrace_input(ConnectivityTracer, x)\n\nEnumerates input indices and constructs the specified type of tracer. Supports JacobianTracer and ConnectivityTracer.\n\nExample\n\njulia> x = rand(3);\n\njulia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];\n\njulia> xt = trace_input(ConnectivityTracer, x)\n3-element Vector{ConnectivityTracer}:\n ConnectivityTracer(1,)\n ConnectivityTracer(2,)\n ConnectivityTracer(3,)\n\njulia> yt = f(xt)\n3-element Vector{ConnectivityTracer}:\n   ConnectivityTracer(1,)\n ConnectivityTracer(1, 2)\n   ConnectivityTracer(3,)\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Utility to extract input indices from tracers:","category":"page"},{"location":"api/","page":"API Reference","title":"API Reference","text":"inputs","category":"page"},{"location":"api/#SparseConnectivityTracer.inputs","page":"API Reference","title":"SparseConnectivityTracer.inputs","text":"inputs(tracer)\n\nReturn raw UInt64 input indices of a ConnectivityTracer or JacobianTracer\n\nExample\n\njulia> t = tracer(ConnectivityTracer, 1, 2, 4)\nConnectivityTracer(1, 2, 4)\n\njulia> inputs(t)\n3-element Vector{Int64}:\n 1\n 2\n 4\n\n\n\n\n\n","category":"function"},{"location":"#SparseConnectivityTracer.jl","page":"Home","title":"SparseConnectivityTracer.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Stable) (Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Aqua)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Fast sparsity detection via operator-overloading.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Will soon include Hessian sparsity detection (#20).","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install this package, open the Julia REPL and run ","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]add SparseConnectivityTracer","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"julia> using SparseConnectivityTracer\n\njulia> x = rand(3);\n\njulia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];\n\njulia> pattern(f, JacobianTracer, x)\n3×3 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 4 stored entries:\n 1  ⋅  ⋅\n 1  1  ⋅\n ⋅  ⋅  1","category":"page"},{"location":"","page":"Home","title":"Home","text":"As a larger example, let's compute the sparsity pattern from a convolutional layer from Flux.jl:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using SparseConnectivityTracer, Flux\n\njulia> x = rand(28, 28, 3, 1);\n\njulia> layer = Conv((3, 3), 3 => 8);\n\njulia> pattern(layer, JacobianTracer, x)\n5408×2352 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 146016 stored entries:\n⎡⠙⢦⡀⠀⠀⠘⢷⣄⠀⠀⠈⠻⣦⡀⠀⠀⠀⎤\n⎢⠀⠀⠙⢷⣄⠀⠀⠙⠷⣄⠀⠀⠈⠻⣦⡀⠀⎥\n⎢⢶⣄⠀⠀⠙⠳⣦⡀⠀⠈⠳⢦⡀⠀⠈⠛⠂⎥\n⎢⠀⠙⢷⣄⠀⠀⠈⠻⣦⡀⠀⠀⠙⢦⣄⠀⠀⎥\n⎢⣀⡀⠀⠉⠳⣄⡀⠀⠈⠻⣦⣀⠀⠀⠙⢷⡄⎥\n⎢⠈⠻⣦⡀⠀⠈⠛⢦⡀⠀⠀⠙⢷⣄⠀⠀⠀⎥\n⎢⠀⠀⠈⠻⣦⡀⠀⠀⠙⢷⣄⠀⠀⠙⠷⣄⠀⎥\n⎢⠻⣦⡀⠀⠈⠙⢷⣄⠀⠀⠉⠻⣦⡀⠀⠈⠁⎥\n⎢⠀⠀⠙⢦⣀⠀⠀⠙⢷⣄⠀⠀⠈⠻⣦⡀⠀⎥\n⎢⢤⣄⠀⠀⠙⠳⣄⡀⠀⠉⠳⣤⡀⠀⠈⠛⠂⎥\n⎢⠀⠙⢷⣄⠀⠀⠈⠻⣦⡀⠀⠈⠙⢦⡀⠀⠀⎥\n⎢⣀⠀⠀⠙⢷⣄⡀⠀⠈⠻⣦⣀⠀⠀⠙⢷⡄⎥\n⎢⠈⠳⣦⡀⠀⠈⠻⣦⡀⠀⠀⠙⢷⣄⠀⠀⠀⎥\n⎢⠀⠀⠈⠻⣦⡀⠀⠀⠙⢦⣄⠀⠀⠙⢷⣄⠀⎥\n⎢⠻⣦⡀⠀⠈⠙⢷⣄⠀⠀⠉⠳⣄⡀⠀⠉⠁⎥\n⎢⠀⠈⠛⢦⡀⠀⠀⠙⢷⣄⠀⠀⠈⠻⣦⡀⠀⎥\n⎢⢤⣄⠀⠀⠙⠶⣄⠀⠀⠙⠷⣤⡀⠀⠈⠻⠆⎥\n⎢⠀⠙⢷⣄⠀⠀⠈⠳⣦⡀⠀⠈⠻⣦⡀⠀⠀⎥\n⎣⠀⠀⠀⠙⢷⣄⠀⠀⠈⠻⣦⠀⠀⠀⠙⢦⡀⎦","category":"page"},{"location":"","page":"Home","title":"Home","text":"SparseConnectivityTracer enumerates inputs x and primal outputs y = f(x) and returns a sparse matrix C of size m times n, where C[i, j] is true if the compute graph connects the j-th entry in x to the i-th entry in y.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For more detailled examples, take a look at the documentation.","category":"page"},{"location":"#Related-packages","page":"Home","title":"Related packages","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SparseDiffTools.jl: automatic sparsity detection via Symbolics.jl and Cassette.jl\nSparsityTracing.jl: automatic Jacobian sparsity detection using an algorithm based on SparsLinC by Bischof et al. (1996)","category":"page"}]
}