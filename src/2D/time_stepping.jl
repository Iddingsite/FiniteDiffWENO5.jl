"""
    WENO_step!(u::T,
               v::NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractArray{<:Real}, 2}}},
               weno::WENOScheme,
               Δt, Δx, Δy) where {T <: AbstractArray{<:Real, 2}}

Advance the solution `u` by one time step using the 3rd-order SSP Runge-Kutta method with WENO5-Z as the spatial discretization in 2D.

# Arguments
- `u::T`: Current solution array to be updated in place.
- `v::NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractArray{<:Real}, 2}}}`: Velocity array (can be staggered or not based on `weno.stag`).
- `weno::WENOScheme`: WENO scheme structure containing necessary parameters and temporary arrays.
- `Δt`: Time step size.
- `Δx`: Spatial grid size in the x-direction.
- `Δy`: Spatial grid size in the y-direction.

Citation: Borges et al. 2008: "An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws"
          doi:10.1016/j.jcp.2007.11.038
"""
function WENO_step!(u::T, v::NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractArray{<:Real}, 2}}}, weno::WENOScheme, Δt, Δx, Δy) where {T <: Array{<:Real, 2}}

    nx, ny = size(u, 1), size(u, 2)
    Δx_, Δy_ = inv(Δx), inv(Δy)

    @unpack ut, du, stag, fl, fr, multithreading = weno

    WENO_flux!(fl, fr, u, weno, nx, ny)
    semi_discretisation_weno5!(du, v, weno, Δx_, Δy_)

    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut)
        ut[I] = @muladd u[I] - Δt * du[I]
    end

    WENO_flux!(fl, fr, ut, weno, nx, ny)
    semi_discretisation_weno5!(du, v, weno, Δx_, Δy_)

    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut)
        ut[I] = @muladd 0.75 * u[I] + 0.25 * ut[I] - 0.25 * Δt * du[I]
    end

    WENO_flux!(fl, fr, ut, weno, nx, ny)
    semi_discretisation_weno5!(du, v, weno, Δx_, Δy_)

    return @inbounds @maybe_threads multithreading for I in CartesianIndices(u)
        u[I] = @muladd 1.0 / 3.0 * u[I] + 2.0 / 3.0 * ut[I] - (2.0 / 3.0) * Δt * du[I]
    end
end
