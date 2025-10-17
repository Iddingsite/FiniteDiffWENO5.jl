
@kernel inbounds = true function WENO_flux_KA_1D(fl, fr, u, boundary, nx, χ, γ, ζ, ϵ)

    I = @index(Global, NTuple)
    I = I
    i = I[1]

    # Left boundary condition
    if boundary[1] == 0       # Dirichlet
        iwww = clamp(i - 3, 1, nx)
        iww = clamp(i - 2, 1, nx)
        iw = clamp(i - 1, 1, nx)
    elseif boundary[1] == 1   # Neumanna
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

    u1 = u[iwww]
    u2 = u[iww]
    u3 = u[iw]
    u4 = u[ie]
    u5 = u[iee]
    u6 = u[ieee]

    fl[i] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
    fr[i] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)
end

@kernel inbounds = true function WENO_semi_discretisation_weno5_KA_1D!(du, fl, fr, v, stag, Δx_)

    I = @index(Global, NTuple)
    I = I
    i = I[1]

    if stag
        du[i] = @muladd (
            max(v.x[i + 1], 0) * fl.x[i + 1] +
                min(v.x[i + 1], 0) * fr.x[i + 1] -
                max(v.x[i], 0) * fl.x[i] -
                min(v.x[i], 0) * fr.x[i]
        ) * Δx_
    else
        du[i] = @muladd max(v.x[i], 0) * (fl.x[i + 1] - fl.x[i]) * Δx_ + min(v.x[i], 0) * (fr.x[i + 1] - fr.x[i]) * Δx_
    end
end

"""
    WENO_step!(u::T_KA, v::NamedTuple{names, <:Tuple{<:T_KA}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, backend::Backend) where T_KA <: AbstractField{<:Real} where names

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 1D.

# Arguments
- `u::T_field`: The current solution field to be updated in place.
- `v::NamedTuple{names, <:Tuple{<:T_field}}`: The velocity field (can be staggered or not based on `weno.stag`). Needs to be a NamedTuple with field `:x`.
- `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and fields.
- `Δt`: The time step size.
- `Δx`: The spatial grid size.
- `backend::Backend`: The KernelAbstractions backend in use (e.g., CPU(), CUDABackend(), etc.).
"""
function WENO_step!(u::T_KA, v::NamedTuple{(:x,), <:Tuple{<:T_KA}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, backend::Backend) where {T_KA <: AbstractVector{<:Real}}

    @assert get_backend(u) == backend
    @assert get_backend(v.x) == backend

    #! do things here for halos and such for clusters for boundaries probably

    @unpack ut, du, fl, fr, stag, boundary, χ, γ, ζ, ϵ = weno

    nx = size(u, 1)
    Δx_ = inv(Δx)

    kernel_flux_1D = WENO_flux_KA_1D(backend)
    kernel_semi_discretisation_1D = WENO_semi_discretisation_weno5_KA_1D!(backend)

    kernel_flux_1D(fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, ndrange = length(fl.x))
    kernel_semi_discretisation_1D(du, fl, fr, v, stag, Δx_, ndrange = length(du))

    ut .= @muladd u .- Δt .* du

    kernel_flux_1D(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, ndrange = length(fl.x))
    kernel_semi_discretisation_1D(du, fl, fr, v, stag, Δx_, ndrange = length(du))

    ut .= @muladd 0.75 .* u .+ 0.25 .* ut .- 0.25 .* Δt .* du

    kernel_flux_1D(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, ndrange = length(fl.x))
    kernel_semi_discretisation_1D(du, fl, fr, v, stag, Δx_, ndrange = length(du))

    u .= @muladd inv(3.0) .* u .+ 2.0 / 3.0 .* ut .- 2.0 / 3.0 .* Δt .* du

    return nothing
end