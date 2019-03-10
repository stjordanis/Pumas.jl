using PuMaS, Test, CSV, Random, Distributions, LinearAlgebra, TransformVariables

theopp = process_nmtran(example_nmtran_data("event_data/THEOPP"),[:WT,:SEX])

theopmodel_bayes = @model begin
    @param begin
      θ ~ Constrained(MvNormal([1.9,0.0781,0.0463,1.5,0.4],
                               Diagonal([9025, 15.25, 5.36, 5625, 400])),
                      lower=[0.1,0.008,0.0004,0.1,0.0001],
                      upper=[5,0.5,0.09,5,1.5],
                      init=[1.9,0.0781,0.0463,1.5,0.4]
                      )
      Ω ~ InverseWishart(2, fill(0.9,1,1) .* (2 + 1 + 1)) # NONMEM specifies the inverse Wishart in terms of its mode
      σ ∈ RealDomain(lower=0.0, init=0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = (SEX == 1 ? θ[1] : θ[4]) + η[1]
      K = θ[2]
      CL  = θ[3]*(WT/70)^θ[5]
      V = CL/K
      SC = V/(WT/70)
    end

    @covariates SEX WT

    @vars begin
        conc = Central / SC
    end

    @dynamics OneCompartmentModel

    @derived begin
        dv ~ @. Normal(conc,sqrt(σ))
    end
end

Random.seed!(1)
try
  b = PuMaS.fit(theopmodel_bayes, theopp, PuMaS.BayesMCMC(),
                nsamples = 10_000)

  m = PuMaS.param_mean(b)

  @test m.θ[1] ≈ 1.72E+00 rtol=0.1
  @test m.θ[2] ≈ 8.57E-02 rtol=0.1
  @test m.θ[3] ≈ 3.90E-02 rtol=0.1
  @test m.θ[4] ≈ 1.73E+00 rtol=0.1
  @test m.θ[5] ≈ 4.32E-01 rtol=0.1
  @test_broken m.Ω[1] ≈ 1.09E+00 rtol=0.1
  @test m.σ ≈ 1.24E+00 rtol=0.1

  s = PuMaS.param_std(b)
  @test s.θ[1] ≈ 4.62E-01 rtol=0.2
  @test s.θ[2] ≈ 6.92E-03 rtol=0.1
  @test s.θ[3] ≈ 2.16E-03 rtol=0.1
  @test s.θ[4] ≈ 4.74E-01 rtol=0.2
  @test s.θ[5] ≈ 1.54E-01 rtol=0.1
  @test_broken s.Ω[1] ≈ 7.00E-01 rtol=0.1
  @test s.σ ≈  1.68E-01 rtol=0.1
catch e
  show(e)
end


theopmodel_bayes2 = @model begin
    @param begin
      θ ~ Constrained(MvNormal([1.9,0.0781,0.0463,1.5,0.4],
                               Diagonal([9025, 15.25, 5.36, 5625, 400])),
                      lower=[0.1,0.008,0.0004,0.1,0.0001],
                      upper=[5,0.5,0.09,5,1.5],
                      init=[1.9,0.0781,0.0463,1.5,0.4]
                      )
      Ω ~ InverseWishart(2, fill(0.9,1,1) .* (2 + 1 + 1)) # NONMEM specifies the inverse Wishart in terms of its mode
      σ ∈ RealDomain(lower=0.0, init=0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = (SEX == 1 ? θ[1] : θ[4]) + η[1]
      K = θ[2]
      CL  = θ[3]*(WT/70)^θ[5]
      V = CL/K
      SC = V/(WT/70)
    end

    @covariates SEX WT

    @vars begin
        conc = Central / SC
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - K*Central
    end

    @derived begin
        dv ~ @. Normal(conc,sqrt(σ))
    end
end

Random.seed!(1)
try
  b = PuMaS.fit(theopmodel_bayes2, theopp, PuMaS.BayesMCMC(),
                nsamples = 10_000)

  m = PuMaS.param_mean(b)

  @test m.θ[1] ≈ 1.72E+00 rtol=0.1
  @test m.θ[2] ≈ 8.57E-02 rtol=0.1
  @test m.θ[3] ≈ 3.90E-02 rtol=0.1
  @test m.θ[4] ≈ 1.73E+00 rtol=0.1
  @test m.θ[5] ≈ 4.32E-01 rtol=0.1
  @test_broken m.Ω[1] ≈ 1.09E+00 rtol=0.1
  @test m.σ ≈ 1.24E+00 rtol=0.1

  s = PuMaS.param_std(b)
  @test s.θ[1] ≈ 4.62E-01 rtol=0.2
  @test s.θ[2] ≈ 6.92E-03 rtol=0.1
  @test s.θ[3] ≈ 2.16E-03 rtol=0.1
  @test s.θ[4] ≈ 4.74E-01 rtol=0.2
  @test s.θ[5] ≈ 1.54E-01 rtol=0.1
  @test_broken s.Ω[1] ≈ 7.00E-01 rtol=0.1
  @test s.σ ≈  1.68E-01 rtol=0.1
catch e
  show(e)
end
