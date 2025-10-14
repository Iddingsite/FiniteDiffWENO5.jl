module FiniteDiffWENO5

using UnPack
using MuladdMacro
using KernelAbstractions
using Chmy

include("utils.jl")
include("WENO5/cache.jl")
include("WENO5/reconstruction.jl")
include("1D/semi_discretisation_1D.jl")
include("1D/time_stepping.jl")
include("2D/semi_discretisation_2D.jl")
include("2D/time_stepping.jl")
include("3D/semi_discretisation_3D.jl")
include("3D/time_stepping.jl")

export WENOScheme, WENO_step!

end # module FiniteDiffWENO5
