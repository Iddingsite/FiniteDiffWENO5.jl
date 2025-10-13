
"""
    WENO_step!(u::T, v, weno::WENOScheme, Δt, Δx, Δy) where T <: AbstractArray{<:Real, 2}

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization in 2D.

# Arguments
- `u::T`: The current solution array to be updated in place.
- `v`: The velocity array (can be staggered or not based on `weno.stag`).
- `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and arrays.
- `Δt`: The time step size.
- `Δx`: The spatial grid size in the x-direction.
- `Δy`: The spatial grid size in the y-direction.
"""
function WENO_step!(u::T, v, weno::WENOScheme, Δt, Δx, Δy) where T <: AbstractArray{<:Real, 2}

    nx, ny = size(u, 1), size(u, 2)
    Δx_, Δy_ = inv(Δx), inv(Δy)

    @unpack ut, du, stag, fl, fr, multithreading = weno

    WENO_flux!(fl, fr, u, weno, nx, ny)
    semi_discretisation_weno5!(du, v, weno, Δx_, Δy_)

    @inbounds @maybe_threads multithreading for I = CartesianIndices(ut)
        ut[I] = @muladd u[I] - Δt * du[I]
    end

    WENO_flux!(fl, fr, ut, weno, nx, ny)
    semi_discretisation_weno5!(du, v, weno, Δx_, Δy_)

    @inbounds @maybe_threads multithreading for I = CartesianIndices(ut)
        ut[I] = @muladd 0.75 * u[I] + 0.25 * ut[I] - 0.25 * Δt * du[I]
    end

    WENO_flux!(fl, fr, ut, weno, nx, ny)
    semi_discretisation_weno5!(du, v, weno, Δx_, Δy_)

    @inbounds @maybe_threads multithreading for I = CartesianIndices(u)
        u[I] = @muladd 1.0/3.0 * u[I] + 2.0/3.0 * ut[I] - (2.0/3.0) * Δt * du[I]
    end
end