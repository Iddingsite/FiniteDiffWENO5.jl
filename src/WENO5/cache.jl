abstract type AbstractWENO end

@kwdef struct WENOScheme{T, N} <: AbstractWENO
    # upwind and downwind constants
    γ::NTuple{3, T} = T.((0.1, 0.6, 0.3))
    # betas' constants
    χ::NTuple{2, T} = T.((13/12, 1/4))
    # stencil weights
    ζ::NTuple{5, T} = T.((1/3, 7/6, 11/6, 1/6, 5/6))
    # tolerance
    ϵ::T = eps(T)
    # conservative
    conservative::Bool
    # WENO-Z (Borges et al. 2008)
    weno_z::Bool
    # fluxes as NamedTuples
    fl::NamedTuple
    fr::NamedTuple
    # semi-discretisation of the advection term
    du::Array{T, N}
    # temporary array for the time stepping
    ut::Array{T, N}
end

"""
    WENOScheme(u0::AbstractArray{T, N}) where {T, N}

Structure containing the Weighted Essentially Non-Oscillatory (WENO) scheme of order 5 constants and arrays for N-dimensional data of type T.

# Fields
- `γ::NTuple{3, T}`: Upwind and downwind constants.
- `χ::NTuple{2, T}`: Betas' constants.
- `ζ::NTuple{5, T}`: Stencil weights.
- `ϵ::T`: Tolerance, fixed to machine precision.
- `fl::NamedTuple`: Fluxes in the left direction for each dimension.
- `fr::NamedTuple`: Fluxes in the right direction for each dimension.
- `du::Array{T, N}`: Semi-discretisation of the advection term.
- `ut::Array{T, N}`: Temporary array for intermediate calculations using Runge-Kutta.
"""
function WENOScheme(u0::AbstractArray{T, N}; conservative::Bool=true, weno_z::Bool=true) where {T, N}

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

    return WENOScheme{T, N}(conservative=conservative, weno_z=weno_z, fl=fl, fr=fr, du=du, ut=ut)
end
