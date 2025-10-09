

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

function WENO_flux!(fl, fr, u, weno, nx)
    @unpack boundary, χ, γ, ζ, ϵ = weno

    bL = Val(boundary[1])
    bR = Val(boundary[2])

    @inbounds for i in 1:nx+1
        iwww = left_index(i, 3, nx, bL)
        iww  = left_index(i, 2, nx, bL)
        iw   = left_index(i, 1, nx, bL)
        ie   = right_index(i, 0, nx, bR)
        iee  = right_index(i, 1, nx, bR)
        ieee = right_index(i, 2, nx, bR)

        u1 = u[iwww]
        u2 = u[iww]
        u3 = u[iw]
        u4 = u[ie]
        u5 = u[iee]
        u6 = u[ieee]

        fl.x[i] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr.x[i] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)
    end
end


function semi_discretisation_weno5!(du::T, v, weno::WENOScheme, Δx_) where T <: AbstractArray{<:Real, 1}

    @unpack fl, fr, stag = weno

    # use staggered grid or not for the velocities
    if stag
        @inbounds for i = axes(du, 1)
            du[i] = @muladd (
                            max(v[i+1], 0) * fl.x[i+1] +
                            min(v[i+1], 0) * fr.x[i+1] -
                            max(v[i], 0) * fl.x[i] -
                            min(v[i], 0) * fr.x[i]
                            ) * Δx_
        end
    else
        @inbounds for i = axes(du, 1)
            du[i] = @muladd max(v[i], 0) * (fl.x[i+1] - fl.x[i]) * Δx_ + min(v[i], 0) * (fr.x[i+1] - fr.x[i]) * Δx_
        end
    end
end
