using Test, SafeTestsets
using PuMaS, LinearAlgebra, Optim

@time begin
@time @safetestset "Simple Model" begin include("simple_model.jl") end
@time @safetestset "Theophylline NLME.jl" begin include("theop_nlme.jl") end
@time @safetestset "Theophylline" begin include("theophylline.jl") end
@time @safetestset "Wang" begin include("wang.jl") end
end
