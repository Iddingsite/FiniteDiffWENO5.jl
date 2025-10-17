module KAExt
using FiniteDiffWENO5
using MuladdMacro
using UnPack
using KernelAbstractions

import FiniteDiffWENO5: WENO_step!

include("KAExt1D.jl")
include("KAExt2D.jl")



end
