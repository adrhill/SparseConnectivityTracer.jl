using SparseConnectivityTracer
using Documenter

DocMeta.setdocmeta!(SparseConnectivityTracer, :DocTestSetup, :(using SparseConnectivityTracer); recursive=true)

makedocs(;
    modules=[SparseConnectivityTracer],
    authors="Adrian Hill <gh@adrianhill.de>",
    sitename="SparseConnectivityTracer.jl",
    format=Documenter.HTML(;
        canonical="https://adrhill.github.io/SparseConnectivityTracer.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/adrhill/SparseConnectivityTracer.jl",
    devbranch="main",
)
