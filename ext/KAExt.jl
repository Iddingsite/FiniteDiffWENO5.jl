module KAExt
using FiniteDiffWENO5
using MuladdMacro
using UnPack
using KernelAbstractions

import FiniteDiffWENO5: WENOScheme, WENO_step!


"""
WENOScheme(c0::AbstractArray{T, N}, backend::Backend; boundary=(2, 2), stag=true) where {T, N}

Create a WENO scheme structure for the given field `c` using the specified `backend` from KernelAbstractions.jl.

# Arguments
- `c0::AbstractArray{T, N}`: The input field for which the WENO scheme is to be created. Only used to get the type and size.
- `backend::Backend`: The KernelAbstractions backend to be used (e.g., CPU(), CUDA(), etc.).
- `boundary::NTuple{2N, Int}`: A tuple specifying the boundary conditions for each dimension (0: homogeneous Neumann, 1: homogeneous Dirichlet, 2: periodic). Default is periodic (2).
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
"""
function WENOScheme(c0::AbstractArray{T, N}, backend::Backend; boundary::NTuple = (2, 2), stag::Bool = true, kwargs...) where {T, N}

    @assert get_backend(c0) == backend "The type of the input field must match the specified backend."

    # check that boundary conditions are correctly defined
    @assert length(boundary) == 2N "Boundary conditions must be a tuple of length $(2N) for $(N)D data."
    # check that boundary conditions are either 0 (homogeneous Neumann) or 1 (homogeneous Dirichlet) or 2 (periodic)
    @assert all(b in (0, 1, 2) for b in boundary) "Boundary conditions must be either 0 (homogeneous Neumann), 1 (homogeneous Dirichlet) or 2 (periodic)."

    # multithreading is always on in this case
    multithreading = true

    backend = get_backend(c0)

    N_boundary = 2 * N

    # helper to expand size in a given dimension
    @inline function flux_size(d)
        return ntuple(i -> size(c0, i) + (i == d ? 1 : 0), min(N, 3))
    end

    # construct NamedTuples for left and right fluxes
    labels = (:x, :y, :z)[1:min(N, 3)]
    fl = NamedTuple{labels}(ntuple(d -> KernelAbstractions.zeros(backend, T, flux_size(d)), min(N, 3)))
    fr = NamedTuple{labels}(ntuple(d -> KernelAbstractions.zeros(backend, T, flux_size(d)), min(N, 3)))

    du = KernelAbstractions.zeros(backend, T, size(c0))
    ut = KernelAbstractions.zeros(backend, T, size(c0))

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{T, TArray, TFlux, N_boundary}(stag = stag, boundary = boundary, multithreading = multithreading, fl = fl, fr = fr, du = du, ut = ut)
end

include("KAExt1D.jl")
include("KAExt2D.jl")
include("KAExt3D.jl")

end
