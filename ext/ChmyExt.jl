module ChmyExt
using FiniteDiffWENO5
using MuladdMacro
using UnPack
using Chmy
using KernelAbstractions

import FiniteDiffWENO5: WENOScheme, WENO_step!


"""
WENOScheme(u::AbstractField{T, N}, grid; boundary=(2, 2), stag=true) where {T, N}

Create a WENO scheme structure for the given field `u` on the specified `grid` using Chmy.jl.

# Arguments
- `c0::AbstractField{T, N}`: The input field for which the WENO scheme is to be created. Only used to get the type and size.
- `grid::StructuredGrid`: The computational grid.
- `boundary::NTuple{2N, Int}`: A tuple specifying the boundary conditions for each dimension (0: homogeneous Neumann, 1: homogeneous Dirichlet, 2: periodic). Default is periodic (2).
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
"""
function WENOScheme(c0::AbstractField{T, N}, grid::StructuredGrid; boundary::NTuple = (2, 2), stag::Bool = true, kwargs...) where {T, N}

    # check that boundary conditions are correctly defined
    @assert length(boundary) == 2N "Boundary conditions must be a tuple of length $(2N) for $(N)D data."
    # check that boundary conditions are either 0 (homogeneous Neumann) or 1 (homogeneous Dirichlet) or 2 (periodic)
    @assert all(b in (0, 1, 2) for b in boundary) "Boundary conditions must be either 0 (homogeneous Neumann), 1 (homogeneous Dirichlet) or 2 (periodic)."

    # multithreading is always on in this case with chmy.jl
    multithreading = true

    backend = get_backend(c0)

    N_boundary = 2 * N

    fl = VectorField(backend, grid)
    fr = VectorField(backend, grid)
    du = Field(backend, grid, Center())
    ut = Field(backend, grid, Center())

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{T, TArray, TFlux, N_boundary}(stag = stag, boundary = boundary, multithreading = multithreading, fl = fl, fr = fr, du = du, ut = ut)
end

function WENOScheme(c0::AbstractField; kwargs...)
    error(
        """
        You called `WENOScheme(c0)` with a `$(typeof(c0))`, which is a subtype of `AbstractField`.

        To construct a WENO scheme for Chmy.jl fields, you must also provide the computational grid:
            WENOScheme(c0::AbstractField, grid::StructuredGrid; kwargs...)

        Example:
            grid = UniformGrid(arch; origin=(0.0, 0.0), extent=(Lx, Lx), dims=(nx, ny))
            weno = WENOScheme(c0, grid; boundary=(2,2,2,2), stag=false)
        """
    )
end

# I am reimporting the files from the discretisation of KA.jl here.
# I didn't find a better way because @kernel functions are not real functions that we can extend from the base package
# and we can't access an extension from another extension
# but at least we don't have to duplicate physically the code
include("KAExt1D.jl")
include("KAExt2D.jl")
include("KAExt3D.jl")


"""
    WENO_step!(u::T_field, v::NamedTuple{names, <:Tuple{<:T_field}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, grid::StructuredGrid, arch) where T_field <: AbstractField{<:Real} where names

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 1D.

# Arguments
- `u::T_field`: The current solution field to be updated in place.
- `v::NamedTuple{names, <:Tuple{<:T_field}}`: The velocity field (can be staggered or not based on `weno.stag`). Needs to be a NamedTuple with field `:x`.
- `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and fields.
- `Δt`: The time step size.
- `Δx`: The spatial grid size.
- `grid::StructuredGrid`: The computational grid.
"""
function WENO_step!(u::T_field, v::NamedTuple{(:x,), <:Tuple{<:T_field}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, grid::StructuredGrid, arch) where {T_field <: AbstractVector{<:Real}}

    launch = Launcher(arch, grid)

    #! do things here for halos and such for clusters for boundaries probably

    nx = grid.axes[1].length
    Δx_ = inv(Δx)

    @unpack ut, du, fl, fr, stag, boundary, χ, γ, ζ, ϵ = weno

    launch(arch, grid, WENO_flux_KA_1D => (fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_semi_discretisation_weno5_KA_1D! => (du, fl, fr, v, stag, Δx_, grid))

    interior(ut) .= @muladd interior(u) .- Δt .* interior(du)

    launch(arch, grid, WENO_flux_KA_1D => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_semi_discretisation_weno5_KA_1D! => (du, fl, fr, v, stag, Δx_, grid))

    interior(ut) .= @muladd 0.75 .* interior(u) .+ 0.25 .* interior(ut) .- 0.25 .* Δt .* interior(du)

    launch(arch, grid, WENO_flux_KA_1D => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_semi_discretisation_weno5_KA_1D! => (du, fl, fr, v, stag, Δx_, grid))

    interior(u) .= @muladd inv(3.0) .* interior(u) .+ 2.0 / 3.0 .* interior(ut) .- 2.0 / 3.0 .* Δt .* interior(du)

    return nothing
end


"""
    WENO_step!(u::T_field, v, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, grid::StructuredGrid, arch) where T_field <: AbstractField{<:Real} where names

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 2D.

# Arguments
- `u::T_field`: The current solution field to be updated in place.
- `v::NamedTuple{names, <:Tuple{<:T_field}}`: The velocity field (can be staggered or not based on `weno.stag`). Needs to be a NamedTuple with fields `:x` and `:y`.
- `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and fields.
- `Δt`: The time step size.
- `Δx`: The spatial grid size.
- `grid::StructuredGrid`: The computational grid.
"""
function WENO_step!(u::T_field, v::NamedTuple{names, <:Tuple{Vararg{AbstractField{<:Real}, 2}}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, grid::StructuredGrid, arch) where {T_field <: AbstractField{<:Real, 2}, names}

    launch = Launcher(arch, grid)

    #! do things here for halos and such for clusters for boundaries probably

    nx = grid.axes[1].length
    ny = grid.axes[2].length
    Δx_ = inv(Δx)
    Δy_ = inv(Δy)

    @unpack ut, du, fl, fr, stag, boundary, χ, γ, ζ, ϵ = weno

    launch(arch, grid, WENO_flux_KA_2D_x => (fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_flux_KA_2D_y => (fl.y, fr.y, u, boundary, ny, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_semi_discretisation_weno5_KA_2D! => (du, fl, fr, v, stag, Δx_, Δy_, grid))

    interior(ut) .= @muladd interior(u) .- Δt .* interior(du)

    launch(arch, grid, WENO_flux_KA_2D_x => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_flux_KA_2D_y => (fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_semi_discretisation_weno5_KA_2D! => (du, fl, fr, v, stag, Δx_, Δy_, grid))

    interior(ut) .= @muladd 0.75 .* interior(u) .+ 0.25 .* interior(ut) .- 0.25 .* Δt .* interior(du)

    launch(arch, grid, WENO_flux_KA_2D_x => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_flux_KA_2D_y => (fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_semi_discretisation_weno5_KA_2D! => (du, fl, fr, v, stag, Δx_, Δy_, grid))

    interior(u) .= @muladd inv(3.0) .* interior(u) .+ 2.0 / 3.0 .* interior(ut) .- 2.0 / 3.0 .* Δt .* interior(du)

    return nothing
end


"""
    WENO_step!(u::T_field, v, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, Δz, grid::StructuredGrid, arch) where T_field <: AbstractArray{<:Real, 3}

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 3D.

# Arguments
- `u::T_field`: The current solution field to be updated in place.
- `v::NamedTuple{names, <:Tuple{<:T_field}}`: The velocity field (can be staggered or not based on `weno.stag`). Needs to be a NamedTuple with fields `:x`, `:y` and `:z`.
- `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and fields.
- `Δt`: The time step size.
- `Δx`: The spatial grid size.
- `Δy`: The spatial grid size.
- `Δz`: The spatial grid size.
- `grid::StructuredGrid`: The computational grid.
"""
function WENO_step!(u::T_field, v, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, Δz, grid::StructuredGrid, arch) where {T_field <: AbstractArray{<:Real, 3}}

    launch = Launcher(arch, grid)

    #! do things here for halos and such for clusters for boundaries probably

    nx = grid.axes[1].length
    ny = grid.axes[2].length
    nz = grid.axes[3].length
    Δx_ = inv(Δx)
    Δy_ = inv(Δy)
    Δz_ = inv(Δz)

    @unpack ut, du, fl, fr, stag, boundary, χ, γ, ζ, ϵ = weno

    launch(arch, grid, WENO_flux_KA_3D_x => (fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_flux_KA_3D_y => (fl.y, fr.y, u, boundary, ny, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_flux_KA_3D_z => (fl.z, fr.z, u, boundary, nz, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_semi_discretisation_weno5_KA_3D! => (du, fl, fr, v, stag, Δx_, Δy_, Δz_, grid))

    interior(ut) .= @muladd interior(u) .- Δt .* interior(du)

    launch(arch, grid, WENO_flux_KA_3D_x => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_flux_KA_3D_y => (fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_flux_KA_3D_z => (fl.z, fr.z, ut, boundary, nz, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_semi_discretisation_weno5_KA_3D! => (du, fl, fr, v, stag, Δx_, Δy_, Δz_, grid))

    interior(ut) .= @muladd 0.75 .* interior(u) .+ 0.25 .* interior(ut) .- 0.25 .* Δt .* interior(du)

    launch(arch, grid, WENO_flux_KA_3D_x => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_flux_KA_3D_y => (fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_flux_KA_3D_z => (fl.z, fr.z, ut, boundary, nz, χ, γ, ζ, ϵ, grid))
    launch(arch, grid, WENO_semi_discretisation_weno5_KA_3D! => (du, fl, fr, v, stag, Δx_, Δy_, Δz_, grid))

    interior(u) .= @muladd inv(3.0) .* interior(u) .+ 2.0 / 3.0 .* interior(ut) .- 2.0 / 3.0 .* Δt .* interior(du)

    return nothing
end


end
