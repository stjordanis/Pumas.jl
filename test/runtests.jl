using PKPDSimulator
using Base.Test

tic()
@time @testset "Single Dosage Tests" begin include("single_dosage_tests.jl") end
@time @testset "Multiple Dosage Tests" begin include("multiple_dosage_tests.jl") end
toc()
