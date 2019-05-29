@time @safetestset "Duplicate Example Check" begin
    include("duplicate_example.jl") end
@time @safetestset "Parsing Tests" begin
    include("parsing_tests.jl") end
@time @safetestset "DSL" begin
    include("dsl.jl") end
@time @safetestset "Mixed PK" begin
    include("mixed_pk.jl") end
@time @safetestset "Parameters" begin
    include("params.jl") end
@time @safetestset "Single Dosage Tests" begin
    include("single_dosage_tests.jl") end
@time @safetestset "Multiple Dosage Tests" begin
    include("multiple_dosage_tests.jl") end
@time @safetestset "Generated Doses Tests" begin
    include("generated_doses_tests.jl") end
@time @safetestset "SS=2 and Overlap Tests" begin
    include("ss2_overlap_tests.jl") end
@time @safetestset "Eventless Tests" begin
    include("eventless.jl") end
@time @safetestset "Template Model EV System" begin
    include("template_model_ev_system.jl") end
@time @safetestset "Multiple Response Tests" begin
    include("multiresponses.jl") end
@time @safetestset "DCP Rate Handling Tests" begin
    include("dcp_rate.jl") end
@time @safetestset "Type-Stability Tests" begin
    include("stability_tests.jl") end
@time @safetestset "StaticArray Tests" begin
     include("static_array_test.jl") end
@time @safetestset "Time-Varying Covariate Tests" begin
     include("time_varying_covar.jl") end
@time @safetestset "Neutropenia" begin
     include("neutropenia.jl") end
@time @safetestset "Error Handling" begin
     include("error_handling.jl") end
@time @safetestset "Automatic Differentiation Tests" begin
     include("ad_tests.jl") end
