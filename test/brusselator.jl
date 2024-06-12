using ADTypes
using ADTypes: AbstractSparsityDetector
using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

const FIRST_ORDER_PATTERNS = (
    IndexSetVectorPattern{Int,BitSet},
    IndexSetVectorPattern{Int,Set{Int}},
    IndexSetVectorPattern{Int,DuplicateVector{Int}},
    IndexSetVectorPattern{Int,RecursiveSet{Int}},
    IndexSetVectorPattern{Int,SortedVector{Int}},
)

include("brusselator_definition.jl")

function test_brusselator(method::AbstractSparsityDetector)
    N = 6
    f! = Brusselator!(N)
    x = rand(N, N, 2)
    y = similar(x)

    J = ADTypes.jacobian_sparsity(f!, y, x, method)
    @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(J)
end

@testset "Pattern type $P" for P in FIRST_ORDER_PATTERNS
    method = TracerSparsityDetector(; first_order=P)
    test_brusselator(method)
end
