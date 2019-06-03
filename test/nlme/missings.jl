using PuMaS, Test, LinearAlgebra

@testset "Test with missing values" begin
  data = process_nmtran(example_nmtran_data("sim_data_model1"))

  # Make a missing observation
  push!(data[1].observations.dv, missing)
  push!(data[1].time, 2)

  model = Dict()

  model["additive"] = @model begin
    @param begin
        θ ∈ RealDomain()
        Ω ∈ PDiagDomain(1)
        σ ∈ RealDomain(lower=0.0001)
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
        dv ~ @. Normal(conc, σ)
    end
  end

  model["proportional"] = @model begin
    @param begin
        θ ∈ RealDomain()
        Ω ∈ PDiagDomain(1)
        σ ∈ RealDomain(lower=0.0001)
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
        dv ~ @. Normal(conc, conc*σ)
    end
  end

  model["exponential"] = @model begin
    @param begin
        θ ∈ RealDomain()
        Ω ∈ PDiagDomain(1)
        σ ∈ RealDomain(lower=0.0001)
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
        dv ~ @. LogNormal(log(conc), σ)
    end
  end

  param = (θ=0.5, Ω=Diagonal([0.04]), σ=0.01)

  @testset "testing model: $_model, with $_approx approximation" for
    _model in ("additive", "proportional", "exponential"),
      _approx in (PuMaS.FO(), PuMaS.FOCE(), PuMaS.FOCEI(), PuMaS.Laplace(), PuMaS.LaplaceI())

    if _model == "proportional" && _approx == PuMaS.LaplaceI()
      # NONMEM also can't handle this one
      @test_throws PosDefException fit(model[_model], data, param, _approx)
    else
      @test fit(model[_model], data, param, _approx) isa PuMaS.FittedPuMaSModel
    end
  end
end