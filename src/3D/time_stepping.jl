
"""
    WENO_step!(u::T, v, weno::WENOScheme, Δt, Δx, Δy, Δz) where T <: AbstractArray{<:Real, 3}

Advance the solution `u` by one time step using the 3rd-order SSP Runge-Kutta method with WENO5-Z as the spatial discretization in 3D.

# Arguments
- `u::T`: The current solution array to be updated in place.
- `v`: The velocity array (can be staggered or not based on `weno.stag`). Needs to be a NamedTuple like (x=..., y=..., z=...).
- `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and temporary arrays.
- `Δt`: The time step size.
- `Δx`: The spatial grid size in the x-direction.
- `Δy`: The spatial grid size in the y-direction.
- `Δz`: The spatial grid size in the z-direction.

Citation: Borges et al. 2008: "An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws"
          doi:10.1016/j.jcp.2007.11.038
"""
function WENO_step!(u::T, v, weno::WENOScheme, Δt, Δx, Δy, Δz) where T <: AbstractArray{<:Real, 3}

    nx, ny, nz = size(u, 1), size(u, 2), size(u, 3)
    Δx_, Δy_, Δz_ = inv(Δx), inv(Δy), inv(Δz)

    @unpack ut, du, stag, fl, fr, multithreading = weno

    WENO_flux!(fl, fr, u, weno, nx, ny, nz)
    semi_discretisation_weno5!(du, v, weno, Δx_, Δy_, Δz_)

    @inbounds @maybe_threads multithreading for I = CartesianIndices(ut)
        ut[I] = @muladd u[I] - Δt * du[I]
    end

    WENO_flux!(fl, fr, ut, weno, nx, ny, nz)
    semi_discretisation_weno5!(du, v, weno, Δx_, Δy_, Δz_)

    @inbounds @maybe_threads multithreading for I = CartesianIndices(ut)
        ut[I] = @muladd 0.75 * u[I] + 0.25 * ut[I] - 0.25 * Δt * du[I]
    end

    WENO_flux!(fl, fr, ut, weno, nx, ny, nz)
    semi_discretisation_weno5!(du, v, weno, Δx_, Δy_, Δz_)

    @inbounds @maybe_threads multithreading for I = CartesianIndices(u)
        u[I] = @muladd 1.0/3.0 * u[I] + 2.0/3.0 * ut[I] - (2.0/3.0) * Δt * du[I]
    end
end