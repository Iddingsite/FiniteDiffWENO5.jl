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

include("ChmyExt1D.jl")
include("ChmyExt2D.jl")
include("ChmyExt3D.jl")


end
