# symb(f) = Symbol(parentmodule(f), '.', nameof(f))
function rootmodule(x)
    parent = parentmodule(x)
    if parent == x
        return parent
    else
        return rootmodule(parent)
    end
end
nameofrootmodule(x) = nameof(rootmodule(x))
