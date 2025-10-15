@inline function left_index(i, d, nx, ::Val{0})
    # Dirichlet (clamped to domain)
    return clamp(i - d, 1, nx)
end

@inline function left_index(i, d, nx, ::Val{1})
    # Neumann (mirror the boundary value)
    return max(i - d, 1)
end

@inline function left_index(i, d, nx, ::Val{2})
    # Periodic (wrap around)
    return mod1(i - d, nx)
end

@inline function right_index(i, d, nx, ::Val{0})
    return clamp(i + d, 1, nx)   # Dirichlet
end

@inline function right_index(i, d, nx, ::Val{1})
    return min(i + d, nx)        # Neumann
end

@inline function right_index(i, d, nx, ::Val{2})
    return mod1(i + d, nx)       # Periodic
end

macro maybe_threads(flag, ex)
    return esc(:(($flag) ? (Base.Threads.@threads $ex) : $ex))
end
