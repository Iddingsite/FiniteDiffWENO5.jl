function WENO_flux!(fl, fr, u, weno, nx)
    @unpack boundary, χ, γ, ζ, ϵ, multithreading = weno

    bL = Val(boundary[1])
    bR = Val(boundary[2])

    return @inbounds @maybe_threads multithreading for i in axes(fl.x, 1)
        iwww = left_index(i, 3, nx, bL)
        iww = left_index(i, 2, nx, bL)
        iw = left_index(i, 1, nx, bL)
        ie = right_index(i, 0, nx, bR)
        iee = right_index(i, 1, nx, bR)
        ieee = right_index(i, 2, nx, bR)

        u1 = u[iwww]
        u2 = u[iww]
        u3 = u[iw]
        u4 = u[ie]
        u5 = u[iee]
        u6 = u[ieee]

        fl.x[i] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr.x[i] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)
    end
end


function semi_discretisation_weno5!(du::T, v, weno::WENOScheme, Δx_) where {T <: AbstractArray{<:Real, 1}}

    @unpack fl, fr, stag, multithreading = weno

    # use staggered grid or not for the velocities
    return if stag
        @inbounds @maybe_threads multithreading for i in axes(du, 1)
            du[i] = @muladd (
                max(v.x[i + 1], 0) * fl.x[i + 1] +
                    min(v.x[i + 1], 0) * fr.x[i + 1] -
                    max(v.x[i], 0) * fl.x[i] -
                    min(v.x[i], 0) * fr.x[i]
            ) * Δx_
        end
    else
        @inbounds @maybe_threads multithreading for i in axes(du, 1)
            du[i] = @muladd max(v.x[i], 0) * (fl.x[i + 1] - fl.x[i]) * Δx_ + min(v.x[i], 0) * (fr.x[i + 1] - fr.x[i]) * Δx_
        end
    end
end
