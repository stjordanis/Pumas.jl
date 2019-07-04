using Test
using Pumas, LinearAlgebra, Optim, CSV

@testset "Models from the Wang paper" begin

  data = read_pumas(example_nmtran_data("wang"))

  # Add initial events following Wang's model
  for i in eachindex(data)
    push!(data[i].events, Pumas.Event(10.0, 0.0, 1, 1))
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

    @test deviance(wang_additive, data, param, Pumas.FO())   ≈  0.026 atol = 1e-3
    @test deviance(wang_additive, data, param, Pumas.FOCE()) ≈ -2.059 atol = 1e-3
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

    @test deviance(wang_proportional, data, param, Pumas.FO())    ≈ 39.213 atol = 1e-3
    @test deviance(wang_proportional, data, param, Pumas.FOCE())  ≈ 39.207 atol = 1e-3
    @test deviance(wang_proportional, data, param, Pumas.FOCEI()) ≈ 39.458 atol = 1e-3
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

    # NONMEM computes the exponential error modeel differently. In stead of deriving the FO/FOCE(I)
    # approximations from the Laplace approximation, NONMEM's version is based on linearization. the
    # two approaches are only equivalent when the modal is Gaussian. Hence we test by log transforming
    # the model

    # First we load a new verison of data and log transform dv
    _df = CSV.read(example_nmtran_data("wang"))
    _df[:dv] = log.(_df[:dv])
    data_log = read_pumas(_df)

    # Add initial events following Wang's model
    for i in eachindex(data_log)
      push!(data_log[i].events, Pumas.Event(10.0, 0.0, 1, 1))
    end

    # Define a model similar to wang_exponential but using Normal instead of LogNormal since
    # dv has been log transformed
    wang_exponential_log = @model begin
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
        dv ~ @. Normal(log(conc), sqrt(σ²))
      end
    end

    # Now use density transformations to test that the two formulations are equivalent
    param = init_param(wang_exponential)

    # Compute the correction term which is sum(log(dg⁻¹(y)/dy)) where g=exp in our case so g⁻¹=log
    correction_term = 2*sum(sum(log.(d.observations.dv)) for d in data)
    @test deviance(wang_exponential, data, param, Pumas.FO())    ≈ deviance(wang_exponential_log, data_log, param, Pumas.FO())    + correction_term
    @test deviance(wang_exponential, data, param, Pumas.FOCE())  ≈ deviance(wang_exponential_log, data_log, param, Pumas.FOCE())  + correction_term
    @test deviance(wang_exponential, data, param, Pumas.FOCEI()) ≈ deviance(wang_exponential_log, data_log, param, Pumas.FOCEI()) + correction_term
  end
end
