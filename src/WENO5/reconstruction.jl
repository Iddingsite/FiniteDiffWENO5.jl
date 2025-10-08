
## Betas
@inline function weno_betas(u1, u2, u3, u4, u5, weno::WENOScheme)
    @unpack χ, ϵ = weno

    β1 = @muladd χ[1] * (u1 - 2 * u2 + u3)^2 + χ[2] * (u1 - 4 * u2 + 3 * u3)^2
    β2 = @muladd χ[1] * (u2 - 2 * u3 + u4)^2 + χ[2] * (u2 - u4)^2
    β3 = @muladd χ[1] * (u3 - 2 * u4 + u5)^2 + χ[2] * (3 * u3 - 4 * u4 + u5)^2

    return β1, β2, β3
end

@inline function weno_alphas_upwind(β1, β2, β3, weno::WENOScheme)
    @unpack γ, ϵ = weno

    # use improved formulation from Borges et al. 2008
    τ = abs(β1 - β3)
    α1L = @muladd γ[1] * (1 + (τ * inv(β1 + ϵ))^2)
    α2L = @muladd γ[2] * (1 + (τ * inv(β2 + ϵ))^2)
    α3L = @muladd γ[3] * (1 + (τ * inv(β3 + ϵ))^2)
    return α1L, α2L, α3L
end

@inline function weno_alphas_downwind(β1, β2, β3, weno::WENOScheme)
    @unpack γ, ϵ = weno

    # use improved formulation from Borges et al. 2008
    τ = abs(β1 - β3)
    α1R = @muladd γ[3] * (1 + (τ * inv(β1 + ϵ))^2)
    α2R = @muladd γ[2] * (1 + (τ * inv(β2 + ϵ))^2)
    α3R = @muladd γ[1] * (1 + (τ * inv(β3 + ϵ))^2)
    return α1R, α2R, α3R
end

## Stencil candidates
@inline function stencil_candidate_upwind(u1, u2, u3, u4, u5, weno::WENOScheme)
    @unpack γ = weno

    s1 = @muladd γ[1] * u1 - γ[2] * u2 + γ[3] * u3
    s2 = @muladd -γ[4] * u2 + γ[5] * u3 + γ[1] * u4
    s3 = @muladd γ[1] * u3 + γ[5] * u4 - γ[4] * u5
    return s1, s2, s3
end

@inline function stencil_candidate_downwind(u1, u2, u3, u4, u5, weno::WENOScheme)
    @unpack γ = weno

    s1 = @muladd -γ[4] * u1 + γ[5] * u2 + γ[1] * u3
    s2 = @muladd γ[1] * u2 + γ[5] * u3 - γ[4] * u4
    s3 = @muladd γ[3] * u3 - γ[2] * u4 + γ[1] * u5
    return s1, s2, s3
end


@inline function weno5(u_view, i, n, weno::WENOScheme, upwind::Bool)

    u1 = u_view[1]
    u2 = u_view[2]
    u3 = u_view[3]
    u4 = u_view[4]
    u5 = u_view[5]

    β1, β2, β3 = weno_betas(u1, u2, u3, u4, u5, weno)

    if upwind
        α1, α2, α3 = weno_alphas_upwind(β1, β2, β3, weno)
        s1, s2, s3 = stencil_candidate_upwind(u1, u2, u3, u4, u5, weno)
    else
        α1, α2, α3 = weno_alphas_downwind(β1, β2, β3, weno)
        s1, s2, s3 = stencil_candidate_downwind(u1, u2, u3, u4, u5, weno)
    end

    _αsum = inv(α0 + α1 + α2)

    ω1 = α1 * _αsum
    ω2 = α2 * _αsum
    ω3 = α3 * _αsum

    f = @muladd ω1 * s1 + ω2 * s2 + ω3 * s3

    return f
end