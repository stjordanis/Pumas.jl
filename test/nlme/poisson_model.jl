using Pumas, Test

@testset "Poisson model" begin

  df = read_pumas(example_nmtran_data("sim_poisson"),cvs = [:dose])


  poisson_model = @model begin
    @param begin
      θ₁ ∈ RealDomain(init=3.0, lower=0.1)
      θ₂ ∈ RealDomain(init=0.5, lower=0.1)
      Ω  ∈ PSDDomain(fill(0.1, 1, 1))
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      baseline = θ₁*exp(η[1])
      d50 = θ₂
    end

    @covariates dose

    @derived begin
      dv ~ @. Poisson(baseline*(1-dose/(dose + d50)))
    end
  end


  param = init_param(poisson_model)
  randeffs = init_randeffs(poisson_model, param)

  @test solve(poisson_model, df[1], param, randeffs) === nothing
  @test simobs(poisson_model, df, param,randeffs) != nothing

  res = simobs(poisson_model, df, param, randeffs)

  initial_estimates = [-8.31130E-01,
                       -9.51865E-01,
                       -1.11581E+00,
                       -7.64425E-01,
                       -7.64425E-01,
                       -6.71273E-01,
                       -8.77897E-01,
                       -1.11581E+00,
                       -9.14281E-01,
                       -1.71009E+00,
                       -1.11581E+00,
                       -6.13244E-01,
                       -1.28933E+00,
                       -9.77623E-01,
                       -7.53683E-01,
                       -1.01738E+00,
                       -1.13057E+00,
                       -6.13244E-01,
                       -1.01738E+00,
                       -4.63717E-01]


  for (i, est) in enumerate(initial_estimates)
    @test (sqrt(param.Ω)*Pumas._orth_empirical_bayes(poisson_model, df[i], param, Pumas.LaplaceI()))[1] ≈ est rtol=1e-5
  end
  @test 2*Pumas.marginal_nll(poisson_model, df, param, Pumas.LaplaceI()) ≈ 4015.70427796336 rtol=1e-3

  o = fit(poisson_model, df, param, Pumas.LaplaceI())
  @test 2*Pumas.marginal_nll(o) ≈ 3809.80599298763 rtol=1e-3

  p = o.param
  @test p.θ₁       ≈ 1.0293E+00 rtol=1e-3
  @test p.θ₂       ≈ 4.5185E-01 rtol=1e-3
  @test p.Ω.mat[1] ≈ 1.2201E-01 rtol=1e-3

  # FO/FOCE(I) not supported for
  @test_throws ArgumentError fit(poisson_model, df, param, Pumas.FO())
  @test_throws ArgumentError fit(poisson_model, df, param, Pumas.FOCE())
  @test_throws ArgumentError fit(poisson_model, df, param, Pumas.FOCEI())
end
