using Test
using PuMaS, LinearAlgebra, Optim

@testset "Models from the Wang paper" begin

  data = process_nmtran(example_nmtran_data("wang"))

  @testset "Check that PuMaS throws when there are no events" begin
    @test_throws ArgumentError PuMaS.timespan(data[1])
  end

  # Add initial events following Wang's model
  for i in eachindex(data)
    push!(data[i].events, PuMaS.Event(10.0, 0.0, 1, 1))
  end

  @testset "Additive error model" begin

    wang_additive = @model begin
      @param begin
        θ  ∈ RealDomain(init=0.5)
        Ω  ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        σ² ∈ RealDomain(init=0.1)
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ * exp(η[1])
        V  = 1.0
      end

      @vars begin
        conc = Central / V
      end

      @dynamics ImmediateAbsorptionModel

      @derived begin
        dv ~ @. Normal(conc, sqrt(σ²))
      end
    end

    param = init_param(wang_additive)

    @test PuMaS.marginal_nll_nonmem(wang_additive, data, param, PuMaS.FO())   ≈  0.026 atol = 1e-3
    @test PuMaS.marginal_nll_nonmem(wang_additive, data, param, PuMaS.FOCE()) ≈ -2.059 atol = 1e-3
  end

  @testset "Proportional error model" begin

    wang_proportional = @model begin
      @param begin
        θ  ∈ RealDomain(init=0.5)
        Ω  ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        σ² ∈ ConstDomain(0.1)
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ * exp(η[1])
        V  = 1.0
      end

      @vars begin
        conc = Central / V
      end

      @dynamics ImmediateAbsorptionModel

      @derived begin
        dv ~ @. Normal(conc, conc*sqrt(σ²))
      end
    end

    param = init_param(wang_proportional)

    @test PuMaS.marginal_nll_nonmem(wang_proportional, data, param, PuMaS.FO())    ≈ 39.213 atol = 1e-3
    @test PuMaS.marginal_nll_nonmem(wang_proportional, data, param, PuMaS.FOCE())  ≈ 39.207 atol = 1e-3
    @test PuMaS.marginal_nll_nonmem(wang_proportional, data, param, PuMaS.FOCEI()) ≈ 39.458 atol = 1e-3
  end

  @testset "Exponential error model" begin
    wang_exponential = @model begin
      @param begin
        θ  ∈ RealDomain(init=0.5)
        Ω  ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        σ² ∈ ConstDomain(0.1)
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ[1] * exp(η[1])
        V  = 1.0
      end

      @vars begin
        conc = Central / V
      end

      @dynamics ImmediateAbsorptionModel

      @derived begin
        dv ~ @. LogNormal(log(conc), sqrt(σ²))
      end
    end

    param = init_param(wang_exponential)

    @test_broken PuMaS.marginal_nll_nonmem(wang_exponential, data, param, PuMaS.FO())    ≈ 39.213 atol = 1e-3
    @test_broken PuMaS.marginal_nll_nonmem(wang_exponential, data, param, PuMaS.FOCE())  ≈ 39.207 atol = 1e-3
    @test_broken PuMaS.marginal_nll_nonmem(wang_exponential, data, param, PuMaS.FOCEI()) ≈ 39.458 atol = 1e-3
  end
end
