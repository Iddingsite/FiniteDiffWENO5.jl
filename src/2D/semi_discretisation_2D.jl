function WENO_flux!(fl, fr, u, weno, nx, ny)
    @unpack boundary, χ, γ, ζ, ϵ, multithreading = weno

    bLx = Val(boundary[1])
    bRx = Val(boundary[2])
    bLy = Val(boundary[3])
    bRy = Val(boundary[4])

    # fusion of loops for better performance
    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.x)
        i, j = Tuple(I)

        iwww = left_index(i, 3, nx, bLx)
        iww = left_index(i, 2, nx, bLx)
        iw = left_index(i, 1, nx, bLx)
        ie = right_index(i, 0, nx, bRx)
        iee = right_index(i, 1, nx, bRx)
        ieee = right_index(i, 2, nx, bRx)

        u1 = u[iwww, j]
        u2 = u[iww, j]
        u3 = u[iw, j]
        u4 = u[ie, j]
        u5 = u[iee, j]
        u6 = u[ieee, j]

        fl.x[I] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr.x[I] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        @inbounds if i < nx + 1
            jwww = left_index(j, 3, ny, bLy)
            jww = left_index(j, 2, ny, bLy)
            jw = left_index(j, 1, ny, bLy)
            je = right_index(j, 0, ny, bRy)
            jee = right_index(j, 1, ny, bRy)
            jeee = right_index(j, 2, ny, bRy)

            u1 = u[i, jwww]
            u2 = u[i, jww]
            u3 = u[i, jw]
            u4 = u[i, je]
            u5 = u[i, jee]
            u6 = u[i, jeee]

            fl.y[I] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
            fr.y[I] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)
        end
    end

    # last column for y
    @inbounds for i in axes(fr.y, 1)
        j = ny + 1

        jwww = left_index(j, 3, ny, bLy)
        jww = left_index(j, 2, ny, bLy)
        jw = left_index(j, 1, ny, bLy)
        je = right_index(j, 0, ny, bRy)
        jee = right_index(j, 1, ny, bRy)
        jeee = right_index(j, 2, ny, bRy)

        u1 = u[i, jwww]
        u2 = u[i, jww]
        u3 = u[i, jw]
        u4 = u[i, je]
        u5 = u[i, jee]
        u6 = u[i, jeee]

        fl.y[i, j] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr.y[i, j] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)
    end

    return nothing
end


function semi_discretisation_weno5!(du::T, v, weno::WENOScheme, Δx_, Δy_) where {T <: AbstractArray{<:Real, 2}}

    @unpack fl, fr, stag, multithreading = weno

    # use staggered grid or not for the velocities
    if stag
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du)

            i, j = Tuple(I)

            du[I] = @muladd (
                max(v.x[i + 1, j], 0) * fl.x[i + 1, j] +
                    min(v.x[i + 1, j], 0) * fr.x[i + 1, j] -
                    max(v.x[I], 0) * fl.x[I] -
                    min(v.x[I], 0) * fr.x[I]
            ) * Δx_ +
                (
                max(v.y[i, j + 1], 0) * fl.y[i, j + 1] +
                    min(v.y[i, j + 1], 0) * fr.y[i, j + 1] -
                    max(v.y[I], 0) * fl.y[I] -
                    min(v.y[I], 0) * fr.y[I]
            ) * Δy_
        end
    else
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du)

            i, j = Tuple(I)

            du[I] = @muladd max(v.x[I], 0) * (fl.x[i + 1, j] - fl.x[I]) * Δx_ +
                min(v.x[I], 0) * (fr.x[i + 1, j] - fr.x[I]) * Δx_ +
                max(v.y[I], 0) * (fl.y[i, j + 1] - fl.y[I]) * Δy_ +
                min(v.y[I], 0) * (fr.y[i, j + 1] - fr.y[I]) * Δy_
        end
    end

    return nothing
end
