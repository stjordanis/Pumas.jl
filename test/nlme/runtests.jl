using Test, SafeTestsets
using PuMaS, LinearAlgebra, Optim, StatsBase

if group == "All" || group == "NLME_NOBAYES"
  @time @safetestset "Non-Bayesian models" begin
    @time @safetestset "Simple Model"                                begin include("simple_model.jl")              end
    @time @safetestset "Simple Model (logistic regression)"          begin include("simple_model_logistic.jl")     end
    @time @safetestset "Simple Model with T-distributed error model" begin include("simple_model_tdist.jl")        end
    @time @safetestset "Simple Model disagnostics"                   begin include("simple_model_diagnostics.jl")  end
    @time @safetestset "Theophylline NLME.jl"                        begin include("theop_nlme.jl")                end
    @time @safetestset "Theophylline"                                begin include("theophylline.jl")              end
    @time @safetestset "Wang"                                        begin include("wang.jl")                      end
    @time @safetestset "Poisson"                                     begin include("poisson_model.jl")             end
    @time @safetestset "Information matrix"                          begin include("information.jl")               end
    @time @safetestset "Bolus"                                       begin include("bolus.jl")                     end
  end
end

if group == "All" || group == "NLME_BAYES"
  @time @safetestset "Bayesian models"                               begin include("bayes.jl")       end
end
