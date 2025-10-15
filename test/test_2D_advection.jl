@testset "2D advection tests" begin
    @testset "2D linear case" begin

        # Number of grid points
        nx = 100
        ny = 100
        Lx = 1.0
        Δx = Lx / nx
        Δy = Lx / ny

        x = range(0, stop = Lx, length = nx)

        # Courant number
        CFL = 0.7
        period = 1

        # Grid x assumed defined
        x = range(0, length = nx, stop = Lx)
        y = range(0, length = ny, stop = Lx)
        grid_array = (x .* ones(ny)', ones(nx) .* y')

        vx0 = ones(nx, ny)
        vy0 = ones(nx, ny)

        v = (; x = vy0, y = vx0)

        x0 = 1 / 4
        c = 0.08

        u0 = zeros(ny, nx)

        for I in CartesianIndices((ny, nx))
            u0[I] = sign(exp(-((grid_array[1][I] - x0)^2 + (grid_array[2][I]' - x0)^2) / c^2) - 0.5) * 0.5 + 0.5
        end

        u = copy(u0)
        weno = WENOScheme(u; boundary = (2, 2, 2, 2), stag = false, multithreading = true)


        # grid size
        Δt = CFL * min(Δx, Δy)^(5 / 3)

        tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)))

        t = 0

        while t < tmax
            WENO_step!(u, v, weno, Δt, Δx, Δy)

            t += Δt

            if t + Δt > tmax
                Δt = tmax - t
            end
        end

        @test isapprox(sum(u), sum(u0); atol = 1.0e-6)
        @test isapprox(maximum(u), maximum(u0); atol = 1.0e-2)
    end

    @testset "2D linear case Chmy CPU" begin

        backend = CPU()
        arch = Arch(backend)

        # Number of grid points
        nx = 100
        ny = 100
        Lx = 1.0
        Δx = Lx / nx
        Δy = Lx / ny

        grid = UniformGrid(arch; origin=(0.0, 0.0), extent=(Lx, Lx), dims=(nx, ny))

        x = range(0, stop = Lx, length = nx)

        # Courant number
        CFL = 0.7
        period = 1

        # Grid x assumed defined
        x = range(0, length = nx, stop = Lx)
        y = range(0, length = ny, stop = Lx)
        grid_array = (x .* ones(ny)', ones(nx) .* y')

        vx0 = ones(nx, ny)
        vy0 = ones(nx, ny)

        v = (; x = vy0, y = vx0)

        x0 = 1 / 4
        c = 0.08

        u0 = zeros(ny, nx)

        for I in CartesianIndices((ny, nx))
            u0[I] = sign(exp(-((grid_array[1][I] - x0)^2 + (grid_array[2][I]' - x0)^2) / c^2) - 0.5) * 0.5 + 0.5
        end

        u = Field(backend, grid, Center())
        set!(u, u0)
        weno = WENOScheme(u, grid; boundary = (2, 2, 2, 2), stag = false, multithreading = true)

        # grid size
        Δt = CFL * min(Δx, Δy)^(5 / 3)

        tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)))

        t = 0
        while t < tmax
            WENO_step!(u, v, weno, Δt, Δx, Δy)

            t += Δt

            if t + Δt > tmax
                Δt = tmax - t
            end
        end
        @test isapprox(sum(u), sum(u0); atol = 1.0e-6)
        @test isapprox(maximum(u), maximum(u0); atol = 1.0e-2)
    end
end
