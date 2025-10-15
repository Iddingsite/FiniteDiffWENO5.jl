using Aqua

@testset "Aqua tests" begin
    Aqua.test_all(
        FiniteDiffWENO5;
        persistent_tasks = false
    )
end
