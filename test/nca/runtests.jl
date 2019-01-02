using PuMaS.NCA, SafeTestsets

@time begin
@time @safetestset "Input Tests" begin include("input_tests.jl") end
@time @safetestset "Single Dosage Tests" begin include("singledose_tests.jl") end
@time @safetestset "Multiple Dosage Tests" begin include("multidose_tests.jl") end
end
