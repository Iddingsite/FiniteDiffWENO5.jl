module ChmyExt1D
    using FiniteDiffWENO5
    using MuladdMacro
    using UnPack
    using Chmy
    using KernelAbstractions

    import FiniteDiffWENO5: WENOScheme, WENO_scheme!


    """
    WENOScheme(u::AbstractField{T, N}, grid; boundary=(2, 2), stag=true, weno_z=false, multithreading=false) where {T, N}

    Create a WENO scheme structure for the given field `u` on the specified `grid` using Chmy.jl.

    # Arguments
    - `u::AbstractField{T, N}`: The input field for which the WENO scheme is to be created.
    - `grid::StructuredGrid`: The computational grid.
    - `boundary::NTuple{2N, Int}`: A tuple specifying the boundary conditions for each dimension (0: homogeneous Neumann, 1: homogeneous Dirichlet, 2: periodic). Default is periodic (2).
    - `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
    - `weno_z::Bool`: Whether to use the WENO-Z formulation (Borges et al. 2008) or not. Default is true.
    """
    function WENOScheme(u::AbstractField{T, N}, grid; boundary=(2, 2), stag=true, weno_z=true) where {T, N}

        # check that boundary conditions are correctly defined
        @assert length(boundary) == 2N "Boundary conditions must be a tuple of length $(2N) for $(N)D data."
        # check that boundary conditions are either 0 (homogeneous Neumann) or 1 (homogeneous Dirichlet) or 2 (periodic)
        @assert all(b in (0, 1, 2) for b in boundary) "Boundary conditions must be either 0 (homogeneous Neumann), 1 (homogeneous Dirichlet) or 2 (periodic)."

        # multithreading is always on in this case with chmy.jl
        multithreading = true

        backend=get_backend(u)

        N_boundary = 2 * N

        fl = VectorField(backend, grid)
        fr = VectorField(backend, grid)
        du = Field(backend, grid, Center())
        ut = Field(backend, grid, Center())

        TFlux = typeof(fl)
        TArray = typeof(du)

        return WENOScheme{T, TArray, TFlux, N_boundary}(stag=stag, boundary=boundary, weno_z=weno_z, multithreading=multithreading, fl=fl, fr=fr, du=du, ut=ut)
    end

    @kernel function WENO_flux_chmy_1D(fl, fr, u, boundary, nx, χ, γ, ζ, ϵ, g::StructuredGrid, O)

        I = @index(Global, NTuple)
        I = I + O
        i = I[1]

        # Left boundary condition
        if boundary[1] == 0       # Dirichlet
            iwww = clamp(i - 3, 1, nx)
            iww  = clamp(i - 2, 1, nx)
            iw   = clamp(i - 1, 1, nx)
        elseif boundary[1] == 1   # Neumann
            iwww = max(i - 3, 1)
            iww  = max(i - 2, 1)
            iw   = max(i - 1, 1)
        elseif boundary[1] == 2   # Periodic
            iwww = mod1(i - 3, nx)
            iww  = mod1(i - 2, nx)
            iw   = mod1(i - 1, nx)
        end

        # Right boundary condition
        if boundary[2] == 0
            ie   = clamp(i, 1, nx)
            iee  = clamp(i + 1, 1, nx)
            ieee = clamp(i + 2, 1, nx)
        elseif boundary[2] == 1
            ie   = min(i, nx)
            iee  = min(i + 1, nx)
            ieee = min(i + 2, nx)
        elseif boundary[2] == 2
            ie   = mod1(i, nx)
            iee  = mod1(i + 1, nx)
            ieee = mod1(i + 2, nx)
        end

        u1 = u[iwww]
        u2 = u[iww]
        u3 = u[iw]
        u4 = u[ie]
        u5 = u[iee]
        u6 = u[ieee]

        fl.x[i] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr.x[i] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)
    end

    @kernel function WENO_semi_discretisation_weno5_chmy!(du, fl, fr, v, stag, Δx_, g::StructuredGrid, O)

        I = @index(Global, NTuple)
        I = I + O
        i = I[1]

        if stag
            du[i] = @muladd (max(v.x[i+1], 0) * fl.x[i + 1] +
                    min(v.x[i+1], 0) * fr.x[i + 1] -
                    max(v.x[i], 0) * fl.x[i] -
                    min(v.x[i], 0) * fr.x[i]
                    ) * Δx_
        else
            du[i] = @muladd max(v[i], 0) * (fl.x[i+1] - fl.x[i]) * Δx_ + min(v[i], 0) * (fr.x[i+1] - fr.x[i]) * Δx_
        end
    end

    """
        WENO_scheme!(u::T_field, v::NamedTuple{names, <:Tuple{<:T_field}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, grid::StructuredGrid, arch) where T_field <: AbstractField{<:Real} where names

    Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields.

    # Arguments
    - `u::T_field`: The current solution field to be updated in place.
    - `v::NamedTuple{names, <:Tuple{<:T_field}}`: The velocity field (can be staggered or not based on `weno.stag`).
    - `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and fields.
    - `Δt`: The time step size.
    - `Δx`: The spatial grid size.
    - `grid::StructuredGrid`: The computational grid.
    """
    function WENO_scheme!(u::T_field, v::NamedTuple{names, <:Tuple{<:T_field}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, grid::StructuredGrid, arch) where T_field <: AbstractVector{<:Real} where names

        backend = get_backend(u)

        launch = Launcher(arch, grid)

        #! do things here for halos and such for clusters maybe?

        nx = grid.axes[1].length
        Δx_ = inv(Δx)

        @unpack ut, du, fl, fr, stag, boundary, χ, γ, ζ, ϵ = weno

        launch(arch, grid, WENO_flux_chmy_1D => (fl, fr, u, boundary, nx, χ, γ, ζ, ϵ, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_chmy! => (du, fl, fr, v, stag, Δx_, grid))

        ut .= @muladd u .- Δt .* du

        launch(arch, grid, WENO_flux_chmy_1D => (fl, fr, ut, boundary, nx, χ, γ, ζ, ϵ, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_chmy! => (du, fl, fr, v, stag, Δx_, grid))

        ut .= @muladd 0.75 .* u .+ 0.25 .* ut .- 0.25 .* Δt .* du

        launch(arch, grid, WENO_flux_chmy_1D => (fl, fr, ut, boundary, nx, χ, γ, ζ, ϵ, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_chmy! => (du, fl, fr, v, stag, Δx_, grid))

        u .= @muladd inv(3.0) .* u .+ 2.0/3.0 .* ut .- 2.0/3.0 .* Δt .* du

        synchronize(backend)
    end


end

