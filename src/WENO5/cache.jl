abstract type AbstractWENO end

@kwdef struct WENOScheme{T, TArray, TFlux, N_boundary} <: AbstractWENO
    # upwind and downwind constants
    γ::NTuple{3, T} = T.((0.1, 0.6, 0.3))
    # betas' constants
    χ::NTuple{2, T} = T.((13/12, 1/4))
    # stencil weights
    ζ::NTuple{5, T} = T.((1/3, 7/6, 11/6, 1/6, 5/6))
    # tolerance to machine precision of the type T
    ϵ::T = eps(T)
    # staggered grid or not (velocities on cell faces or cell centers)
    stag::Bool
    # boundary conditions
    boundary::NTuple{N_boundary, Int}
    # WENO-Z (Borges et al. 2008)
    weno_z::Bool
    # multithreading (only used for 2D and 3D)
    multithreading::Bool
    # fluxes as NamedTuples
    fl::TFlux
    fr::TFlux
    # semi-discretisation of the advection term
    du::TArray
    # temporary array for the time stepping
    ut::TArray
end

"""
    WENOScheme(u0::AbstractArray{T, N}; boundary::NTuple=ntuple(i -> 0, N*2), stag::Bool=false, weno_z::Bool=true,  multithreading::Bool=false) where {T, N}

Structure containing the Weighted Essentially Non-Oscillatory (WENO) scheme of order 5 constants and arrays for N-dimensional data of type T.

# Fields
- `γ::NTuple{3, T}`: Upwind and downwind constants.
- `χ::NTuple{2, T}`: Betas' constants.
- `ζ::NTuple{5, T}`: Stencil weights.
- `ϵ::T`: Tolerance, fixed to machine precision.
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
- `boundary::NTuple{N_boundary, Int}`: Boundary conditions for each dimension (0: homogeneous Neumann, 1: homogeneous Dirichlet, 2: periodic). Default to homogeneous Neumann.
- `weno_z::Bool`: Whether to use the WENO-Z formulation (Borges et al. 2008) or not.
- `multithreading::Bool`: Whether to use multithreading (only for 2D and 3D).
- `fl::NamedTuple`: Fluxes in the left direction for each dimension.
- `fr::NamedTuple`: Fluxes in the right direction for each dimension.
- `du::Array{T, N}`: Semi-discretisation of the advection term.
- `ut::Array{T, N}`: Temporary array for intermediate calculations using Runge-Kutta.
"""
function WENOScheme(u0::AbstractArray{T, N}; boundary::NTuple=ntuple(i -> 0, N*2), stag::Bool=false, weno_z::Bool=true, multithreading::Bool=false) where {T, N}

    # check that boundary conditions are correctly defined
    @assert length(boundary) == 2N "Boundary conditions must be a tuple of length $(2N) for $(N)D data."
    # check that boundary conditions are either 0 (homogeneous Neumann) or 1 (homogeneous Dirichlet) or 2 (periodic)
    @assert all(b in (0, 1, 2) for b in boundary) "Boundary conditions must be either 0 (homogeneous Neumann), 1 (homogeneous Dirichlet) or 2 (periodic)."

    # dimension labels
    labels = (:x, :y, :z)[1:min(N, 3)]
    sizes = size(u0)

    # helper to expand size in a given dimension
    function flux_size(d)
        ntuple(i -> sizes[i] + (i == d ? 1 : 0), min(N, 3))
        # ntuple(i -> sizes[i], N)
    end

    # construct NamedTuples for left and right fluxes
    fl = NamedTuple{labels}(ntuple(d -> zeros(T, flux_size(d)), min(N, 3)))
    fr = NamedTuple{labels}(ntuple(d -> zeros(T, flux_size(d)), min(N, 3)))

    # semi-discretisation array
    du = zeros(T, size(u0))

    # temporary array for Runge-Kutta
    ut = zeros(T, size(u0))

    # boundary conditions tuple length
    N_boundary = 2*N

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{T, TArray, TFlux, N_boundary}(stag=stag, boundary=boundary, weno_z=weno_z, multithreading=multithreading, fl=fl, fr=fr, du=du, ut=ut)
end
