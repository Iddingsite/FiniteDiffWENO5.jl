@testset "reconstruction tests" begin
    @testset "Linear case" begin
        u = [1.0, 2.0, 3.0, 4.0, 5.0]
        weno = WENOScheme(u)

        @unpack χ, γ, ζ, ϵ = weno

        f_up = FiniteDiffWENO5.weno5_reconstruction_upwind(u[1], u[2], u[3], u[4], u[5], χ, γ, ζ, ϵ)
        f_down = FiniteDiffWENO5.weno5_reconstruction_downwind(u[1], u[2], u[3], u[4], u[5], χ, γ, ζ, ϵ)

        @test f_up ≈ 3.5
        @test f_down ≈ 2.5
    end
end
