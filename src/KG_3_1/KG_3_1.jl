module KG_3_1

using Jecco
using LinearAlgebra

import Base.Threads.@threads
import Base.Threads.@spawn

# abstract types and structs used throughout
include("types.jl")

include("system.jl")

include("potential.jl")

# include("initial_data.jl")


# include("param.jl")

# include("dphidt.jl")
# include("equation_coeff.jl")
# include("solve_nested.jl")
# include("rhs.jl")
# include("run.jl")
# include("ibvp.jl")

# export ParamBase, ParamGrid, ParamID, ParamEvol, ParamIO
# export Potential
# export VV # this will contain the potential
# export System
# export BulkVars, BoundaryVars, AllVars


# always set the number of BLAS threads to 1 upon loading the module. by default
# it uses a bunch of them and we don't want that since they trample over each
# other when solving the nested systems equations. it's much better to thread
# over the loop. see also the discussion here:
# https://github.com/JuliaLang/julia/issues/33409
#
# this saves us the tedious task of always setting OMP_NUM_THREADS=1 before
# launching julia.
function __init__()
    LinearAlgebra.BLAS.set_num_threads(1)
    nothing
end

export SpecCartGrid3D
export Potential, ConstPotential, SquarePotential
export System, SystemPartition

end
