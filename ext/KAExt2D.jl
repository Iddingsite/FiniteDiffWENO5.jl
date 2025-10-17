@kernel function WENO_flux_KA_2D_x(fl, fr, u, boundary, nx, χ, γ, ζ, ϵ)

    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    n, m = size(fl)

    if 1 <= i <= n && 1 <= j <= m

        # Left boundary condition
        if boundary[1] == 0       # homogeneous Dirichlet
            iwww = clamp(i - 3, 1, nx)
            iww = clamp(i - 2, 1, nx)
            iw = clamp(i - 1, 1, nx)
        elseif boundary[1] == 1   # homogeneous Neumann
            iwww = max(i - 3, 1)
            iww = max(i - 2, 1)
            iw = max(i - 1, 1)
        elseif boundary[1] == 2   # Periodic
            iwww = mod1(i - 3, nx)
            iww = mod1(i - 2, nx)
            iw = mod1(i - 1, nx)
        end

        # Right boundary condition
        if boundary[2] == 0
            ie = clamp(i, 1, nx)
            iee = clamp(i + 1, 1, nx)
            ieee = clamp(i + 2, 1, nx)
        elseif boundary[2] == 1
            ie = min(i, nx)
            iee = min(i + 1, nx)
            ieee = min(i + 2, nx)
        elseif boundary[2] == 2
            ie = mod1(i, nx)
            iee = mod1(i + 1, nx)
            ieee = mod1(i + 2, nx)
        end

        u1 = u[iwww, j]
        u2 = u[iww, j]
        u3 = u[iw, j]
        u4 = u[ie, j]
        u5 = u[iee, j]
        u6 = u[ieee, j]

        fl[i, j] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)
    end
end


@kernel function WENO_flux_KA_2D_y(fl, fr, u, boundary, ny, χ, γ, ζ, ϵ)

    I = @index(Global, NTuple)
    i, j = I[1], I[2]
    n, m = size(fl)

    if 1 <= i <= n && 1 <= j <= m

        # Left boundary condition
        if boundary[3] == 0       # homogeneous Dirichlet
            jwww = clamp(j - 3, 1, ny)
            jww = clamp(j - 2, 1, ny)
            jw = clamp(j - 1, 1, ny)
        elseif boundary[3] == 1   # homogeneous Neumann
            jwww = max(j - 3, 1)
            jww = max(j - 2, 1)
            jw = max(j - 1, 1)
        elseif boundary[3] == 2   # Periodic
            jwww = mod1(j - 3, ny)
            jww = mod1(j - 2, ny)
            jw = mod1(j - 1, ny)
        end

        # Right boundary condition
        if boundary[4] == 0
            je = clamp(j, 1, ny)
            jee = clamp(j + 1, 1, ny)
            jeee = clamp(j + 2, 1, ny)
        elseif boundary[4] == 1
            je = min(j, ny)
            jee = min(j + 1, ny)
            jeee = min(j + 2, ny)
        elseif boundary[4] == 2
            je = mod1(j, ny)
            jee = mod1(j + 1, ny)
            jeee = mod1(j + 2, ny)
        end

        u1 = u[i, jwww]
        u2 = u[i, jww]
        u3 = u[i, jw]
        u4 = u[i, je]
        u5 = u[i, jee]
        u6 = u[i, jeee]

        fl[i, j] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)
    end
end

@kernel function WENO_semi_discretisation_weno5_KA_2D!(du, fl, fr, v, stag, Δx_, Δy_)

    I = @index(Global, Cartesian)

    i, j = I[1], I[2]

    m, n = size(du)

    if 1 <= i <= m && 1 <= j <= n
        if stag
            du[I] = @muladd (
                max(v.x[i + 1, j], 0) * fl.x[i + 1, j] +
                    min(v.x[i + 1, j], 0) * fr.x[i + 1, j] -
                    max(v.x[I], 0) * fl.x[I] -
                    min(v.x[I], 0) * fr.x[I]
            ) * Δx_ +
                (
                max(v.y[i, j + 1], 0) * fl.y[i, j + 1] +
                    min(v.y[i, j + 1], 0) * fr.y[i, j + 1] -
                    max(v.y[I], 0) * fl.y[I] -
                    min(v.y[I], 0) * fr.y[I]
            ) * Δy_
        else
            du[I] = @muladd max(v.x[I], 0) * (fl.x[i + 1, j] - fl.x[I]) * Δx_ + min(v.x[I], 0) * (fr.x[i + 1, j] - fr.x[I]) * Δx_ +
                max(v.y[I], 0) * (fl.y[i, j + 1] - fl.y[I]) * Δy_ + min(v.y[I], 0) * (fr.y[i, j + 1] - fr.y[I]) * Δy_
        end
    end
end

"""
    WENO_step!(u::T_field, v, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, backend::Backend) where T_field <: AbstractField{<:Real} where names

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 2D.

# Arguments
- `u::T_KA`: The current solution field to be updated in place.
- `v::NamedTuple{names, <:Tuple{<:T_KA}}`: The velocity field (can be staggered or not based on `weno.stag`). Needs to be a NamedTuple with fields `:x` and `:y`.
- `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and fields.
- `Δt`: The time step size.
- `Δx`: The spatial grid size.
- `Δy`: The spatial grid size.
- `backend::Backend`: The KernelAbstractions backend in use (e.g., CPU(), CUDABackend(), etc.).
"""
function WENO_step!(u::T_KA, v, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, backend::Backend) where {T_KA <: AbstractArray{<:Real, 2}}

    @assert get_backend(u) == backend
    @assert get_backend(v.x) == backend
    @assert get_backend(v.y) == backend

    #! do things here for halos and such for clusters for boundaries probably

    nx = size(u, 1)
    ny = size(u, 2)
    Δx_ = inv(Δx)
    Δy_ = inv(Δy)

    @unpack ut, du, fl, fr, stag, boundary, χ, γ, ζ, ϵ = weno

    flx_l = size(fl.x)
    fly_l = size(fl.y)
    du_l = size(du)

    kernel_flux_2D_x = WENO_flux_KA_2D_x(backend)
    kernel_flux_2D_y = WENO_flux_KA_2D_y(backend)
    kernel_semi_discretisation_2D = WENO_semi_discretisation_weno5_KA_2D!(backend)

    kernel_flux_2D_x(fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, ndrange = flx_l)
    kernel_flux_2D_y(fl.y, fr.y, u, boundary, ny, χ, γ, ζ, ϵ, ndrange = fly_l)
    kernel_semi_discretisation_2D(du, fl, fr, v, stag, Δx_, Δy_, ndrange = du_l)

    ut .= @muladd u .- Δt .* du


    kernel_flux_2D_x(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, ndrange = flx_l)
    kernel_flux_2D_y(fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, ndrange = fly_l)
    kernel_semi_discretisation_2D(du, fl, fr, v, stag, Δx_, Δy_, ndrange = du_l)

    ut .= @muladd 0.75 .* u .+ 0.25 .* ut .- 0.25 .* Δt .* du

    kernel_flux_2D_x(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, ndrange = flx_l)
    kernel_flux_2D_y(fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, ndrange = fly_l)
    kernel_semi_discretisation_2D(du, fl, fr, v, stag, Δx_, Δy_, ndrange = du_l)

    u .= @muladd inv(3.0) .* u .+ 2.0 / 3.0 .* ut .- 2.0 / 3.0 .* Δt .* du

    return nothing
end
