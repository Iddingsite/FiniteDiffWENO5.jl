module ChmyExt
using FiniteDiffWENO5
using MuladdMacro
using UnPack
using Chmy
using KernelAbstractions

import FiniteDiffWENO5: WENOScheme, WENO_step!


"""
WENOScheme(u::AbstractField{T, N}, grid; boundary=(2, 2), stag=true, multithreading=false) where {T, N}

Create a WENO scheme structure for the given field `u` on the specified `grid` using Chmy.jl.

# Arguments
- `u::AbstractField{T, N}`: The input field for which the WENO scheme is to be created.
- `grid::StructuredGrid`: The computational grid.
- `boundary::NTuple{2N, Int}`: A tuple specifying the boundary conditions for each dimension (0: homogeneous Neumann, 1: homogeneous Dirichlet, 2: periodic). Default is periodic (2).
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
- `multithreading::Bool`: Whether to use multithreading (only for 2D and 3D). Default is false.
"""
function WENOScheme(u::AbstractField{T, N}, grid; boundary = (2, 2), stag = true) where {T, N}

    # check that boundary conditions are correctly defined
    @assert length(boundary) == 2N "Boundary conditions must be a tuple of length $(2N) for $(N)D data."
    # check that boundary conditions are either 0 (homogeneous Neumann) or 1 (homogeneous Dirichlet) or 2 (periodic)
    @assert all(b in (0, 1, 2) for b in boundary) "Boundary conditions must be either 0 (homogeneous Neumann), 1 (homogeneous Dirichlet) or 2 (periodic)."

    # multithreading is always on in this case with chmy.jl
    multithreading = true

    backend = get_backend(u)

    N_boundary = 2 * N

    fl = VectorField(backend, grid)
    fr = VectorField(backend, grid)
    du = Field(backend, grid, Center())
    ut = Field(backend, grid, Center())

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{T, TArray, TFlux, N_boundary}(stag = stag, boundary = boundary, multithreading = multithreading, fl = fl, fr = fr, du = du, ut = ut)
end

include("ChmyExt1D.jl")
include("ChmyExt2D.jl")
include("ChmyExt3D.jl")


end
