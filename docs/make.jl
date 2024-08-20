using SparseConnectivityTracer
using Documenter
using DocumenterMermaid

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
        assets    = ["assets/favicon.ico"],
    ),
    pages=[
        "Getting Started" => "index.md",
        "User Documentation" => ["API Reference" => "user/api.md"],
        "Developer Documentation" => [
            "How SCT works" => "dev/how_it_works.md",
            "Internals Reference" => "dev/api.md",
        ],
    ],
    warnonly=[:missing_docs],
)

deploydocs(; repo="github.com/adrhill/SparseConnectivityTracer.jl", devbranch="main")
