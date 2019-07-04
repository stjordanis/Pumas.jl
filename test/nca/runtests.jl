using Pumas.NCA, SafeTestsets

@time begin
@time @safetestset "Input Tests" begin include("input_tests.jl") end
@time @safetestset "Single Dosage Tests" begin include("singledose_tests.jl") end
@time @safetestset "Multiple Dosage Tests" begin include("multidose_tests.jl") end
@time @safetestset "Grouping Tests" begin include("group_tests.jl") end
@time @safetestset "Infusion Tests" begin include("infusion_tests.jl") end
@time @safetestset "Pumas Integration Tests" begin include("pumas_tests.jl") end
@time @safetestset "Urine Tests" begin include("urine_tests.jl") end
end
