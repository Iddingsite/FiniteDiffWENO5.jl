
"""
    WENO_step!(u::T, v, weno::WENOScheme, Δt, Δx) where T <: AbstractVector{<:Real}

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization.

# Arguments
- `u::T`: The current solution array to be updated in place.
- `v`: The velocity array (can be staggered or not based on `weno.stag`).
- `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and arrays.
- `Δt`: The time step size.
- `Δx`: The spatial grid size.
"""
function WENO_step!(u::T, v, weno::WENOScheme, Δt, Δx) where T <: AbstractVector{<:Real}

    nx = size(u, 1)
    Δx_ = inv(Δx)

    @unpack ut, du, stag, fl, fr = weno

    WENO_flux!(fl, fr, u, weno, nx)
    semi_discretisation_weno5!(du, v, weno, Δx_)

    @inbounds for i = axes(ut, 1)
        ut[i] = @muladd u[i] - Δt * du[i]
    end

    WENO_flux!(fl, fr, ut, weno, nx)
    semi_discretisation_weno5!(du, v, weno, Δx_)

    @inbounds for i = axes(ut, 1)
        ut[i] = @muladd 0.75 * u[i] + 0.25 * ut[i] - 0.25 * Δt * du[i]
    end

    WENO_flux!(fl, fr, ut, weno, nx)
    semi_discretisation_weno5!(du, v, weno, Δx_)

    @inbounds for i = axes(u, 1)
        u[i] = @muladd 1.0/3.0 * u[i] + 2.0/3.0 * ut[i] - (2.0/3.0) * Δt * du[i]
    end

end