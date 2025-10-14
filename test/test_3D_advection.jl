@testset "struct tests" begin
    @testset "3D linear case" begin

        nx = 50
        ny = 50
        nz = 50

        L = 1.0
        Δx = L / nx
        Δy = L / ny
        Δz = L / nz

        x = range(0, stop=L, length=nx)
        y = range(0, stop=L, length=ny)
        z = range(0, stop=L, length=nz)

        # Courant number
        CFL = 0.7
        period = 1

        # 3D grid
        X = reshape(x, 1, 1, nx) .* ones(ny, nz, 1)
        Y = reshape(y, 1, ny, 1) .* ones(nx, 1, nz)
        Z = reshape(z, nz, 1, 1) .* ones(1, ny, nx)

        vx0 = ones(size(X))
        vy0 = ones(size(Y))
        vz0 = zeros(size(X)) # Rotation in XY plane only

        v = (; x=vx0, y=vy0, z=vz0)

        x0 = 1/4
        c = 0.08

        u0 = zeros(ny, nx, nz)
        for I in CartesianIndices((ny, nx, nz))
            u0[I] = exp(-((X[I]-x0)^2 + (Y[I]-x0)^2 + (Z[I]-x0)^2) / c^2)
        end

        u = copy(u0)
        weno = WENOScheme(u; boundary=(2, 2, 2, 2, 2, 2), stag=false, multithreading=true)

        Δt = CFL * min(Δx, Δy, Δz)^(5/3)
        tmax = period * L / max(maximum(abs.(vx0)), maximum(abs.(vy0)), maximum(abs.(vz0)))
        t = 0
        counter = 0

        while t < tmax
            WENO_step!(u, v, weno, Δt, Δx, Δy, Δz)

            t += Δt
            if t + Δt > tmax
                Δt = tmax - t
            end

            counter += 1
        end

        @test isapprox(sum(u), sum(u0); atol=1e-6)
        @test maximum(u) ≈ 0.9541260266954649 atol=1e-8
    end

    @testset "3D linear case Chmy CPU" begin

        backend=CPU()
        arch = Arch(backend)

        nx = 50
        ny = 50
        nz = 50

        Lx = 1.0
        Δx = Lx / nx
        Δy = Lx / ny
        Δz = Lx / nz

        grid = UniformGrid(arch; origin=(0.0, 0.0, 0.0), extent=(Lx, Lx, Lx), dims=(nx, ny, nz))

        # Courant number
        CFL = 0.7
        period = 1

        # 3D grid
        x = range(0, length=nx, stop=Lx)
        y = range(0, length=ny, stop=Lx)
        z = range(0, length=nz, stop=Lx)

        X = reshape(x, nx, 1, 1)
        Y = reshape(y, 1, ny, 1)
        Z = reshape(z, 1, 1, nz)

        X3D = X .+ 0 .* Y .+ 0 .* Z
        Y3D = 0 .* X .+ Y .+ 0 .* Z
        Z3D = 0 .* X .+ 0 .* Y .+ Z

        vx0 = ones(size(X3D))
        vy0 = ones(size(Y3D))
        vz0 = zeros(size(Z3D)) # Rotation in XY plane only

        v = (; x=vx0, y=vy0, z=vz0)

        x0 = 1/4
        c = 0.08

        u0 = zeros(ny, nx, nz)
        for I in CartesianIndices((ny, nx, nz))
            u0[I] = exp(-((X3D[I]-x0)^2 + (Y3D[I]-x0)^2 + (Z3D[I]-x0)^2) / c^2)
        end

        u = copy(u0)
        weno = WENOScheme(u; boundary=(2, 2, 2, 2, 2, 2), stag=false, multithreading=true)

        Δt = CFL * min(Δx, Δy, Δz)^(5/3)
        tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)), maximum(abs.(vz0)))
        t = 0
        counter = 0

        while t < tmax
            WENO_step!(u, v, weno, Δt, Δx, Δy, Δz)

            t += Δt
            if t + Δt > tmax
                Δt = tmax - t
            end

            counter += 1
        end

        @test isapprox(sum(u), sum(u0); atol=1e-6)
        @test maximum(u) ≈ 0.9541260266954649 atol=1e-8
    end
end
