@time @safetestset "Duplicate Example Check" begin
    include("core/duplicate_example.jl") end
@time @safetestset "Parsing Tests" begin
    include("core/parsing_tests.jl") end
@time @safetestset "DSL" begin
    include("core/dsl.jl") end
@time @safetestset "Parameters" begin
    include("core/params.jl") end
@time @safetestset "Single Dosage Tests" begin
    include("core/single_dosage_tests.jl") end
@time @safetestset "Multiple Dosage Tests" begin
    include("core/multiple_dosage_tests.jl") end
@time @safetestset "Generated Doses Tests" begin
    include("core/generated_doses_tests.jl") end
@time @safetestset "SS=2 and Overlap Tests" begin
    include("core/ss2_overlap_tests.jl") end
@time @safetestset "Template Model EV System" begin
    include("core/template_model_ev_system.jl") end
@time @safetestset "Multiple Response Tests" begin
    include("core/multiresponses.jl") end
@time @safetestset "Type-Stability Tests" begin
    include("core/stability_tests.jl") end
@time @safetestset "StaticArray Tests" begin
     include("core/static_array_test.jl") end
@time @safetestset "Time-Varying Covariate Tests" begin
     include("core/time_varying_covar.jl") end
@time @safetestset "Neutropenia" begin
     include("core/neutropenia.jl") end
@time @safetestset "Error Handling" begin
     include("core/error_handling.jl") end
@time @safetestset "Automatic Differentiation Tests" begin
     include("core/ad_tests.jl") end
