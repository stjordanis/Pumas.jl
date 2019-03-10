@time @safetestset "SDE Tests" begin
    include("features/sdes.jl") end
@time @safetestset "DDE Tests" begin
    include("features/ddes.jl") end
@time @safetestset "Unit Handling Tests" begin
    include("features/unit_handling_tests.jl") end
@time @safetestset "Physical Measurements Tests" begin
    include("features/measurement_tests.jl") end
#=
@time @safetestset "Discrete Stochastic Tests" begin
     include("features/discrete_stochastic.jl") end
@time @safetestset "Mixed ODE and Discrete Tests" begin
     include("features/mixed_ode_discrete.jl") end
=#
