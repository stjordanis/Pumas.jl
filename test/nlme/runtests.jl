using Test
using PuMaS, LinearAlgebra, Optim

@time begin
@time @safetestset "Simple Model" begin include("simple_model.jl") end
@time @safetestset "Theophylline NLME.jl" begin include("theop_nlme.jl") end
@time @safetestset "Theophylline" begin include("theophylline.jl") end
end