using Pumas, LinearAlgebra

@testset "Test informationmatrix with warfarin data" begin

  data = read_pumas(example_nmtran_data("warfarin"))

  model = @model begin

    @param begin
      θ₁ ∈ RealDomain(lower=0.0, init=0.15)
      θ₂ ∈ RealDomain(lower=0.0, init=8.0)
      θ₃ ∈ RealDomain(lower=0.0, init=1.0)
      Ω  ∈ PDiagDomain(3)
      σ  ∈ RealDomain(lower=0.0001, init=0.01)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Tvcl = θ₁
      Tvv  = θ₂
      Tvka = θ₃
      CL   = Tvcl*exp(η[1])
      V    = Tvv*exp(η[2])
      Ka   = Tvka*exp(η[3])
    end

    @dynamics OneCompartmentModel

    @vars begin
      conc = Central/V
    end

    @derived begin
      dv ~ @. Normal(log(conc), sqrt(σ))
    end
  end

  param = (θ₁ = 0.15,
           θ₂ = 8.0,
           θ₃ = 1.0,
           Ω  = Diagonal([0.07, 0.02, 0.6]),
           σ  = 0.01)

  @test logdet(
    sum(
      Pumas._expected_information(
        model,
        d,
        param,
        Pumas._orth_empirical_bayes(model, d, param, Pumas.FO()),
        Pumas.FO()
      ) for d in data)) ≈ 53.8955 rtol=1e-6

  ft = fit(model, data, param, Pumas.FO())

  @test logdet(informationmatrix(ft)) isa Number

end
