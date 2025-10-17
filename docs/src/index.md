# [FiniteDiffWENO5.jl](@id home)

FiniteDiffWENO5.jl is a Julia package that implements fifth-order finite-difference weighted essentially non-oscillatory (WENO) schemes for solving hyperbolic partial differential equations (PDEs) in 1D, 2D and 3D on regular grids.

The package currently focuses on the non-conservative form of the advection terms ($\mathbf{v} \cdot \nabla u$) on a collocated grid, and the conservative form ($\nabla \cdot$ ($\mathbf{v} u$)) where the velocity field $\mathbf{v}$ and scalar field $u$ are on a staggered grid with the advection velocity located on the sides of the cells.

The core of the package is written in pure Julia, focusing on performance using CPUs, but GPU support is available using KernelAbstractions.jl and Chmy.jl via an extension.

## Installation

FiniteDiffWENO5.jl is a registered package and may be installed directly with the following command in the Julia REPL

```julia-repl
julia>]
  pkg> add FiniteDiffWENO5
  pkg> test FiniteDiffWENO5
```
