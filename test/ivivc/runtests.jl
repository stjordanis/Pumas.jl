using SafeTestsets

@time begin
@time @safetestset "Data parsing Tests" begin include("data_parsing_tests.jl") end
@time @safetestset "Deconvolution Methods Tests" begin include("deconvo_methods_tests.jl") end
end