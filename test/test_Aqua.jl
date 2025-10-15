using Aqua

@testset "Aqua tests" begin
  Aqua.test_all(
    FiniteDiffWENO5;
  )
end