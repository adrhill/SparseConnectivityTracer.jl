using ComponentArrays
using SparseConnectivityTracer
using Test

f(x::AbstractVector) = abs2.(x)
f_comp(x::ComponentVector) = ComponentVector(; a = abs2.(x.a), b = abs2.(x.b))

function f!(y::AbstractVector, x::AbstractVector)
    y .= abs2.(x)
    return y
end

function f_comp!(y::ComponentVector, x::ComponentVector)
    y.a .= abs2.(x.a)
    y.b .= abs2.(x.b)
    return y
end

x_comp = ComponentVector(; a = rand(2), b = rand(3))
y_comp = ComponentVector(; a = rand(2), b = rand(3))
x = Vector(x_comp)
y = Vector(y_comp)

@testset "$detector" for detector in
    (TracerSparsityDetector(), TracerLocalSparsityDetector())
    @test jacobian_sparsity(f_comp, x_comp, detector) == jacobian_sparsity(f, x, detector)
    @test jacobian_sparsity(f_comp!, similar(y_comp), x_comp, detector) ==
        jacobian_sparsity(f!, similar(y), x, detector)
    @test hessian_sparsity(sum ∘ f_comp, x_comp, detector) ==
        hessian_sparsity(sum ∘ f, x, detector)
end
