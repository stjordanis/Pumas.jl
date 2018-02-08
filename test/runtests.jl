using PKPDSimulator
using Base.Test

tic()
@time @testset "Parsing Tests" begin include("parsing_tests.jl") end
@time @testset "Single Dosage Tests" begin include("single_dosage_tests.jl") end
@time @testset "Multiple Dosage Tests" begin include("multiple_dosage_tests.jl") end
@time @testset "SS=2 and Overlap Tests" begin
                include("ss2_overlap_tests.jl") end
@time @testset "Template Model EV System" begin
                include("template_model_ev_system.jl") end
@time @testset "Multiple Response Tests" begin
                include("multiresponses.jl") end
@time @testset "Analytical Type-Stability Tests" begin
                include("analytical_stability_tests.jl") end
@time @testset "ODE Type-Stability Tests" begin
                include("ode_stability_tests.jl") end
@time @testset "StaticArray Tests" begin
                include("static_array_test.jl") end
toc()
