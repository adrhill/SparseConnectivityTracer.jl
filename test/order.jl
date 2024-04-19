using ForwardDiff: derivative, gradient, hessian
using SparseConnectivityTracer
using Test

@enum Order zero_order = 0 first_order = 1 second_order = 2 no_order = 1000

function parse_order(c::Char)
    if c == 'z'
        return zero_order
    elseif c == 'f'
        return first_order
    elseif c == 's' || c == 'c'
        return second_order
    end
end

function parse_orders(code::Symbol)
    str = string(code)
    return Tuple(parse_order(str[i]) for i in eachindex(str))
end

@test parse_orders(:z) == (zero_order,)
@test parse_orders(:fs) == (first_order, second_order)
@test parse_orders(:sfc) == (second_order, first_order, second_order)

## 1 to 1

second_derivative(f, x) = derivative(_x -> derivative(f, _x), x)

function classify_1_to_1(f, x; atol)
    d1 = derivative(f, x)
    d2 = second_derivative(f, x)
    if abs(d1) <= atol && abs(d2) <= atol
        return (zero_order,)
    elseif abs(d1) > atol && abs(d2) <= atol
        return (first_order,)
    elseif abs(d1) > atol && abs(d2) > atol
        return (second_order,)
    end
end

function classify_1_to_1(f; atol=1e-5)
    try
        return classify_1_to_1(f, rand(); atol)
    catch e
        try
            return classify_1_to_1(f, 1 + rand(); atol)
        catch e
            @warn "Classification failed" e
            return (no_order,)
        end
    end
end

@testset verbose = true "1-to-1" begin
    @testset "$code" for code in (:s, :f, :z)
        assigned_order = parse_orders(code)
        list = @eval SparseConnectivityTracer.$(Symbol("ops_1_to_1_$code"))
        for f_symb in list
            numerical_order = classify_1_to_1(eval(f_symb))
            @test (f_symb, numerical_order) <= (f_symb, assigned_order)
        end
    end
end;

## 2 to 1

function classify_2_to_1(f, x, y; atol)
    g = gradient(Base.splat(f), [x, y])
    H = hessian(Base.splat(f), [x, y])
    first_arg = if abs(g[1]) <= atol && abs(H[1, 1]) <= atol
        zero_order
    elseif abs(g[1]) > atol && abs(H[1, 1]) <= atol
        first_order
    elseif abs(g[1]) > atol && abs(H[1, 1]) > atol
        second_order
    end
    second_arg = if abs(g[2]) <= atol && abs(H[2, 2]) <= atol
        zero_order
    elseif abs(g[2]) > atol && abs(H[2, 2]) <= atol
        first_order
    elseif abs(g[2]) > atol && abs(H[2, 2]) > atol
        second_order
    end
    cross = if abs(H[1, 2]) <= atol
        zero_order
    else
        second_order
    end
    return (first_arg, second_arg, cross)
end

function classify_2_to_1(f; atol=1e-5)
    try
        return classify_2_to_1(f, rand(), rand(); atol)
    catch e
        @warn "Classification failed" e
        return (no_order, no_order, no_order)
    end
end

@testset verbose = true "2-to-1" begin
    @testset "$code" for code in (
        :ssc, :ssz, :sfc, :sfz, :fsc, :fsz, :ffc, :ffz, :szz, :zsz, :fzz, :zfz, :zzz
    )
        assigned_order = parse_orders(code)
        list = @eval SparseConnectivityTracer.$(Symbol("ops_2_to_1_$code"))
        for f_symb in list
            numerical_order = classify_2_to_1(eval(f_symb))
            @test (f_symb, numerical_order) <= (f_symb, assigned_order)
        end
    end
end;

## 1 to 2

function classify_1_to_2(f, x; atol)
    d1 = derivative(f, x)
    d2 = second_derivative(f, x)
    first_arg = if abs(d1[1]) <= atol && abs(d2[1]) <= atol
        zero_order
    elseif abs(d1[1]) > atol && abs(d2[1]) <= atol
        first_order
    elseif abs(d1[1]) > atol && abs(d2[1]) > atol
        second_order
    end
    second_arg = if abs(d1[2]) <= atol && abs(d2[2]) <= atol
        zero_order
    elseif abs(d1[2]) > atol && abs(d2[2]) <= atol
        first_order
    elseif abs(d1[2]) > atol && abs(d2[2]) > atol
        second_order
    end
    return (first_arg, second_arg)
end

function classify_1_to_2(f; atol=1e-5)
    try
        return classify_1_to_1(f, rand(); atol)
    catch e
        @warn "Classification failed" e
        return (no_order, no_order)
    end
end

@testset verbose = true "1-to-2" begin
    @testset "$code" for code in (:sf, :sz, :fs, :ff, :fz)
        assigned_order = parse_orders(code)
        list = @eval SparseConnectivityTracer.$(Symbol("ops_1_to_2_$code"))
        for f_symb in list
            numerical_order = classify_1_to_2(eval(f_symb))
            @test (f_symb, numerical_order) <= (f_symb, assigned_order)
        end
    end
end;
