using SparseConnectivityTracer
using Documenter

# Create index.md from README
cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

DocMeta.setdocmeta!(
    SparseConnectivityTracer,
    :DocTestSetup,
    :(using SparseConnectivityTracer);
    recursive=true,
)

makedocs(;
    modules=[SparseConnectivityTracer],
    authors="Adrian Hill <gh@adrianhill.de>",
    sitename="SparseConnectivityTracer.jl",
    format=Documenter.HTML(;
        canonical = "https://adrhill.github.io/SparseConnectivityTracer.jl",
        edit_link = "main",
        assets    = String[],
    ),
    pages=["Home" => "index.md", "API Reference" => "api.md"],
)

deploydocs(; repo="github.com/adrhill/SparseConnectivityTracer.jl", devbranch="main")
