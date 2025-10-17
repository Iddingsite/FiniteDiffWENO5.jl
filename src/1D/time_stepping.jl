"""
    WENO_step!(u::T,
               v::NamedTuple{(:x,), <:Tuple{<:Vector{<:Real}}},
               weno::WENOScheme,
               Δt, Δx) where T <: AbstractVector{<:Real}

Advance the solution `u` by one time step using the 3rd-order SSP Runge-Kutta method with WENO5-Z as the spatial discretization in 1D.

# Arguments
- `u::T`: Current solution array to be updated in place.
- `v::NamedTuple{(:x,), <:Tuple{<:Vector{<:Real}}}`: Velocity array (can be staggered or not based on `weno.stag`).
- `weno::WENOScheme`: WENO scheme structure containing necessary parameters and temporary arrays.
- `Δt`: Time step size.
- `Δx`: Spatial grid size.

Citation: Borges et al. 2008: "An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws"
          doi:10.1016/j.jcp.2007.11.038
"""
function WENO_step!(u::T, v::NamedTuple{(:x,), <:Tuple{<:Vector{<:Real}}}, weno::WENOScheme, Δt, Δx) where {T <: Vector{<:Real}}

    nx = size(u, 1)
    Δx_ = inv(Δx)

    @unpack ut, du, stag, fl, fr, multithreading = weno

    WENO_flux!(fl, fr, u, weno, nx)
    semi_discretisation_weno5!(du, v, weno, Δx_)

    @inbounds @maybe_threads multithreading for i in axes(ut, 1)
        ut[i] = @muladd u[i] - Δt * du[i]
    end

    WENO_flux!(fl, fr, ut, weno, nx)
    semi_discretisation_weno5!(du, v, weno, Δx_)

    @inbounds @maybe_threads multithreading for i in axes(ut, 1)
        ut[i] = @muladd 0.75 * u[i] + 0.25 * ut[i] - 0.25 * Δt * du[i]
    end

    WENO_flux!(fl, fr, ut, weno, nx)
    semi_discretisation_weno5!(du, v, weno, Δx_)

    @inbounds @maybe_threads multithreading for i in axes(u, 1)
        u[i] = @muladd 1.0 / 3.0 * u[i] + 2.0 / 3.0 * ut[i] - (2.0 / 3.0) * Δt * du[i]
    end

    return nothing
end
