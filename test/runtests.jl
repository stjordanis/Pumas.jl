using PKPDSimulator
using Base.Test

tic()
@time @testset "Parsing Tests" begin include("parsing_tests.jl") end
@time @testset "Single Dosage Tests" begin include("single_dosage_tests.jl") end
@time @testset "Analytical Single Dosage Tests" begin
                include("analytical_single_dosage_tests.jl") end
@time @testset "Multiple Dosage Tests" begin include("multiple_dosage_tests.jl") end
@time @testset "Analytical Multiple Dosage Tests" begin
                include("analytical_multiple_dosage_tests.jl") end
@time @testset "SS=2 and Overlap Tests" begin
                include("ss2_overlap_tests.jl") end
@time @testset "Template Model EV System" begin
                include("template_model_ev_system.jl") end
toc()
