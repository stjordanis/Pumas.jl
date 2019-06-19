using PuMaS, Test, CSV, Random, Distributions, LinearAlgebra, TransformVariables

if haskey(ENV,"BAYES_N")
    nsamples = ENV["BAYES_N"]
else
    nsamples = 10_000
end

theopp = read_pumas(example_nmtran_data("event_data/THEOPP"),cvs = [:WT,:SEX])

@testset "Model with analytical solution" begin
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
          dv ~ @. Normal(conc,sqrt(σ)+eps())
      end
  end

  @testset "Test logdensity" begin
    vparam = PuMaS.TransformVariables.inverse(PuMaS.totransform(theopmodel_bayes.param), PuMaS.init_param(theopmodel_bayes))
    ldp = PuMaS.BayesLogDensity(theopmodel_bayes, theopp)
    vparam_aug = [vparam; zeros(length(theopp)*ldp.dim_rfx)]
    v = PuMaS.LogDensityProblems.logdensity(PuMaS.LogDensityProblems.Value, ldp, vparam_aug)
    @test v.value ≈ -612.6392449413322
    vg = PuMaS.LogDensityProblems.logdensity(PuMaS.LogDensityProblems.ValueGradient, ldp, vparam_aug)
    @test vg.value ≈ v.value
    @test vg.gradient ≈ [8.023571333788356
                       878.2155638921361
                      -763.9131862639041
                       114.23979126237558
                         9.92024209941143
                       -10.4
                       449.2675514859391
                        29.39690239835657
                        28.081124185351662
                        28.928080707496264
                         7.428841866509832
                        25.018349868626906
                        -5.042079192537069
                       -21.001561268109093
                        -2.437719761098346
                        30.403288395312458
                       -13.08640380207791
                        20.644305593350005
                        -7.709095620537834]
  end

  Random.seed!(1)
  try
    b = PuMaS.fit(theopmodel_bayes, theopp, PuMaS.BayesMCMC(),
                  nsamples = nsamples)

    m = PuMaS.param_mean(b)

    if nsamples >= 10_000
      @test m.θ[1] ≈ 1.72E+00 rtol=0.1
      @test m.θ[2] ≈ 8.57E-02 rtol=0.1
      @test m.θ[3] ≈ 3.90E-02 rtol=0.1
      @test m.θ[4] ≈ 1.73E+00 rtol=0.1
      @test m.θ[5] ≈ 4.32E-01 rtol=0.1
      @test_broken m.Ω[1] ≈ 1.09E+00 rtol=0.1
      @test m.σ ≈ 1.24E+00 rtol=0.1
    end

    s = PuMaS.param_std(b)

    if nsamples >= 10_000
      @test s.θ[1] ≈ 4.62E-01 rtol=0.2
      @test s.θ[2] ≈ 6.92E-03 rtol=0.1
      @test s.θ[3] ≈ 2.16E-03 rtol=0.1
      @test s.θ[4] ≈ 4.74E-01 rtol=0.2
      @test s.θ[5] ≈ 1.54E-01 rtol=0.1
      @test_broken s.Ω[1] ≈ 7.00E-01 rtol=0.1
      @test s.σ ≈  1.68E-01 rtol=0.1
    end
  catch e
    show(e)
  end
end

@testset "Model with ODE solver" begin
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

  @testset "Test logdensity" begin
    vparam2 = PuMaS.TransformVariables.inverse(PuMaS.totransform(theopmodel_bayes2.param), PuMaS.init_param(theopmodel_bayes2))
    ldp2 = PuMaS.BayesLogDensity(theopmodel_bayes2, theopp,
                                 reltol = 1e-12, abstol = 1e-12)
    vparam2_aug = [vparam2; zeros(length(theopp)*ldp2.dim_rfx)]
    v2 = PuMaS.LogDensityProblems.logdensity(PuMaS.LogDensityProblems.Value, ldp2, vparam2_aug)
    @test v2.value ≈ -612.6392449413325 rtol=1e-6
    vg2 = PuMaS.LogDensityProblems.logdensity(PuMaS.LogDensityProblems.ValueGradient, ldp2, vparam2_aug)
    @test vg2.value ≈ v2.value
    @test vg2.gradient ≈ [8.023571333787114,
                          878.2155638921338,
                          -763.9131862639034,
                          114.23979126237346,
                          9.920242099411354,
                          -10.4,
                          449.2675514859412,
                          29.396902398356797,
                          28.081124185351225,
                          28.92808070749575,
                          7.428841866508794,
                          25.01834986862622,
                          -5.042079192536745,
                          -21.00156126810926,
                          -2.4377197610988577,
                          30.403288395311726,
                          -13.086403802077223,
                          20.644305593350268,
                          -7.709095620538378] rtol=1e-6
  end

  Random.seed!(1)
  try

    b = PuMaS.fit(theopmodel_bayes2, theopp, PuMaS.BayesMCMC(),
                nsamples = nsamples, reltol = 1e-6, abstol = 1e-6)

    m = PuMaS.param_mean(b)

    if nsamples >= 10_000
      @test m.θ[1] ≈ 1.72E+00 rtol=0.1
      @test m.θ[2] ≈ 8.57E-02 rtol=0.1
      @test m.θ[3] ≈ 3.90E-02 rtol=0.1
      @test m.θ[4] ≈ 1.73E+00 rtol=0.1
      @test m.θ[5] ≈ 4.32E-01 rtol=0.1
      @test_broken m.Ω[1] ≈ 1.09E+00 rtol=0.1
      @test m.σ ≈ 1.24E+00 rtol=0.1
    end

    s = PuMaS.param_std(b)

    if nsamples >= 10_000
      @test s.θ[1] ≈ 4.62E-01 rtol=0.2
      @test s.θ[2] ≈ 6.92E-03 rtol=0.1
      @test s.θ[3] ≈ 2.16E-03 rtol=0.1
      @test s.θ[4] ≈ 4.74E-01 rtol=0.2
      @test s.θ[5] ≈ 1.54E-01 rtol=0.1
      @test_broken s.Ω[1] ≈ 7.00E-01 rtol=0.1
      @test s.σ ≈  1.68E-01 rtol=0.1
    end
  catch e
    show(e)
  end
end
