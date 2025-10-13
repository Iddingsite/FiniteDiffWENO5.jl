module FiniteDiffWENO5

using UnPack
using MuladdMacro
using KernelAbstractions
using Chmy

include("WENO5/cache.jl")
include("WENO5/reconstruction.jl")
include("1D/semi_discretisation_1D.jl")
include("1D/time_stepping.jl")

export WENOScheme, WENO_step!

end # module FiniteDiffWENO5
