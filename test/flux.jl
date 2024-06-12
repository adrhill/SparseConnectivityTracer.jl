using ADTypes
using ADTypes: AbstractSparsityDetector
using Flux: Conv, relu
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

const INPUT_FLUX = reshape(
    [
        0.2677768300138966
        1.1934917429169245
        -1.0496617141319355
        0.456668782925957
        0.09678342859916624
        -0.7962039825333248
        -0.6138709208787495
        -0.6809396498148278
        0.4938230574627916
        0.7847107012511034
        0.7423059724033608
        -0.6914378396432983
        1.2062310319178624
        -0.19647670394840708
        0.10708057449244994
        -0.4787927739226245
        0.045072020113458774
        -1.219617669693635
    ],
    3,
    3,
    2,
    1,
) # WHCN
const WEIGHTS_FLUX = reshape(
    [
        0.311843398150865
        0.488663701947109
        0.648497438559604
        -0.41742794246238
        0.174865988551499
        1.061745573803265
        -0.72434245370475
        -0.05213963181095
    ],
    2,
    2,
    2,
    1,
)
const BIAS_FLUX = [0.1]

const LAYER = Conv(WEIGHTS_FLUX, BIAS_FLUX) # Conv((2, 2), 2 => 1)
const LAYER_RELU = Conv(WEIGHTS_FLUX, BIAS_FLUX, relu) # Conv((2, 2), 2 => 1, relu)

function test_flux_conv(method::AbstractSparsityDetector)
    J = jacobian_sparsity(LAYER, INPUT_FLUX, method)
    @test_reference "references/pattern/jacobian/NNlib/conv.txt" BitMatrix(J)
end
function test_flux_conv_local(method::AbstractSparsityDetector)
    J = jacobian_sparsity(LAYER_RELU, INPUT_FLUX, method)
    @test_reference "references/pattern/jacobian/NNlib/conv_relu.txt" BitMatrix(J)
end

@testset "Global" begin
    @testset "$P" for P in FIRST_ORDER_PATTERNS
        method = TracerSparsityDetector(; first_order=P)
        test_flux_conv(method)
    end
end
@testset "Local" begin
    @testset "$P" for P in FIRST_ORDER_PATTERNS
        method = TracerLocalSparsityDetector(; first_order=P)
        test_flux_conv(method)
        test_flux_conv_local(method)
    end
end
