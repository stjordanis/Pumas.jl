using Test
using PuMaS, LinearAlgebra, Optim

# FIXME! Find a nicer way to handle this
_extract(A::PDMat) = A.mat
_extract(A::PDiagMat) = Diagonal(A.diag)
_extract(A) = A

@testset "Theophylline model" begin

# Control verbosity of solver output
verbose = false

theopp = process_nmtran(example_nmtran_data("event_data/THEOPP"),[:SEX,:WT])
@testset "Check that Events is fully typed when parsed" begin
  @test theopp[1].events isa Vector{PuMaS.Event{Float64,Float64,Float64,Float64,Float64,Float64,Int}}
end

@testset "run2.mod FO without interaction, diagonal omega and additive error" begin
  #Note: run2 requires a block diagonal for the omega
  #$OMEGA  BLOCK(3)
  # 5.55  ;       KA__
  # 0.00524 0.00024  ;   COV KA~K
  # -0.128 0.00911 0.515  ;         K_
  theopmodel_analytical_fo = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PSDDomain(3)
      σ_add ∈ RealDomain(lower=0.001, init=0.388)
      #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂+ η[2]
      CL = θ₃*WT + η[3]
      V  = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics OneCompartmentModel

    @derived begin
      dv ~ @. Normal(conc,sqrt(σ_add))
    end
  end

  param = (
    θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
    Ω  = diagm(0 => [5.55, 0.0024, 0.515]), # update to block diagonal
    σ_add = 0.388
    # σ_prop = 0.3
       )

  @test deviance(theopmodel_analytical_fo, theopp, param, PuMaS.FO()) ≈ 137.16573310096661

  fo_estimated_params = (θ₁ = 4.20241E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                         θ₂ = 7.25283E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                         θ₃ = 3.57499E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                         θ₄ = 2.12401E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

                         Ω = [1.81195E+01 -1.12474E-02 -3.05266E-02
                             -1.12474E-02  2.25098E-04  1.04586E-02
                             -3.05266E-02  1.04586E-02  5.99850E-01],
                         σ_add = 2.66533E-01)

  fo_stderr           = (θ₁ = 4.47E-01,
                         θ₂ = 6.81E-03,
                         θ₃ = 4.93E-03,
                         θ₄ = 3.52E-01,
                         Ω = [9.04E+00 1.92E-02 1.25E+00
                              1.92E-02 7.86E-05 4.57E-03
                              1.25E+00 4.57E-03 3.16E-01],
                         σ_add = 5.81E-02)
                         # Elapsed estimation time in seconds:     0.04
                         # Elapsed covariance time in seconds:     0.02

  o = fit(theopmodel_analytical_fo, theopp, param, PuMaS.FO())

  o_estimates = o.param
  o_stderror  = stderror(o)

  o_infer = infer(o)

  # Verify that show runs
  io_buffer = IOBuffer()
  show(io_buffer, o)
  show(io_buffer, o_infer)


  @test deviance(o) ≈ 71.979975297638589

  @testset "test estimate of $k" for k in keys(o_estimates)
    @test _extract(getfield(o_estimates, k)) ≈ _extract(getfield(fo_estimated_params, k)) rtol=1e-3
  end

  @testset "test stderror of $k" for k in keys(o_estimates)
    @test _extract(getfield(o_stderror, k))  ≈ _extract(getfield(fo_stderr, k))           rtol=2e-2
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test o.vvrandeffs[i] == zeros(3)
  end

  pred = predict(o)
  pred_foce = predict(o, PuMaS.FOCE())
  pred_focei = predict(o, PuMaS.FOCEI())
  dfpred = DataFrame(pred)
  dfpred_no_covar = DataFrame(pred; include_covariates=false)
  dfpred_foce = DataFrame(pred_foce)
  dfpred_foce_no_covar = DataFrame(pred_foce; include_covariates=false)
  dfpred_focei = DataFrame(pred_focei)
  dfpred_focei_no_covar = DataFrame(pred_focei; include_covariates=false)
  @testset "test predict" begin
    @test pred[1].approx == PuMaS.FO()
    @test pred_foce[1].approx == PuMaS.FOCE()
    @test pred_focei[1].approx == PuMaS.FOCEI()

    @test all(dfpred[:approx].== Ref(PuMaS.FO()))
    @test all(dfpred_no_covar[:approx].== Ref(PuMaS.FO()))
    @test all(dfpred_foce[:approx].== Ref(PuMaS.FOCE()))
    @test all(dfpred_foce_no_covar[:approx].== Ref(PuMaS.FOCE()))
    @test all(dfpred_focei[:approx].== Ref(PuMaS.FOCEI()))
    @test all(dfpred_focei_no_covar[:approx].== Ref(PuMaS.FOCEI()))

    @test haskey(dfpred, :pred)
    @test haskey(dfpred, :ipred)
    @test haskey(dfpred, :SEX)
    @test haskey(dfpred, :WT)
    @test haskey(dfpred_no_covar, :pred)
    @test haskey(dfpred_no_covar, :ipred)
    @test !(haskey(dfpred_no_covar, :SEX))
    @test !(haskey(dfpred_no_covar, :WT))

    @test haskey(dfpred_foce, :pred)
    @test haskey(dfpred_foce, :ipred)
    @test haskey(dfpred_foce, :SEX)
    @test haskey(dfpred_foce, :WT)
    @test haskey(dfpred_foce_no_covar, :pred)
    @test haskey(dfpred_foce_no_covar, :ipred)
    @test !(haskey(dfpred_foce_no_covar, :SEX))
    @test !(haskey(dfpred_foce_no_covar, :WT))

    @test haskey(dfpred_focei, :pred)
    @test haskey(dfpred_focei, :ipred)
    @test haskey(dfpred_focei, :SEX)
    @test haskey(dfpred_focei, :WT)
    @test haskey(dfpred_focei_no_covar, :pred)
    @test haskey(dfpred_focei_no_covar, :ipred)
    @test !(haskey(dfpred_focei_no_covar, :SEX))
    @test !(haskey(dfpred_focei_no_covar, :WT))
  end

  wres = wresiduals(o)
  wres_foce = wresiduals(o, PuMaS.FOCE())
  wres_focei = wresiduals(o, PuMaS.FOCEI())
  dfwres = DataFrame(wres)
  dfwres_no_covar = DataFrame(wres; include_covariates=false)
  dfwres_foce = DataFrame(wres_foce)
  dfwres_foce_no_covar = DataFrame(wres_foce; include_covariates=false)
  dfwres_focei = DataFrame(wres_focei)
  dfwres_focei_no_covar = DataFrame(wres_focei; include_covariates=false)
  @testset "test wresiduals" begin
    @test wres[1].approx == PuMaS.FO()
    @test wres_foce[1].approx == PuMaS.FOCE()
    @test wres_focei[1].approx == PuMaS.FOCEI()

    @test all(dfwres[:approx].== Ref(PuMaS.FO()))
    @test all(dfwres_no_covar[:approx].== Ref(PuMaS.FO()))
    @test all(dfwres_foce[:approx].== Ref(PuMaS.FOCE()))
    @test all(dfwres_foce_no_covar[:approx].== Ref(PuMaS.FOCE()))
    @test all(dfwres_focei[:approx].== Ref(PuMaS.FOCEI()))
    @test all(dfwres_focei_no_covar[:approx].== Ref(PuMaS.FOCEI()))

    @test haskey(dfwres, :wres)
    @test haskey(dfwres, :iwres)
    @test haskey(dfwres, :SEX)
    @test haskey(dfwres, :WT)
    @test haskey(dfwres_no_covar, :wres)
    @test haskey(dfwres_no_covar, :iwres)
    @test !(haskey(dfwres_no_covar, :SEX))
    @test !(haskey(dfwres_no_covar, :WT))

    @test haskey(dfwres_foce, :wres)
    @test haskey(dfwres_foce, :iwres)
    @test haskey(dfwres_foce, :SEX)
    @test haskey(dfwres_foce, :WT)
    @test haskey(dfwres_foce_no_covar, :wres)
    @test haskey(dfwres_foce_no_covar, :iwres)
    @test !(haskey(dfwres_foce_no_covar, :SEX))
    @test !(haskey(dfwres_foce_no_covar, :WT))

    @test haskey(dfwres_focei, :wres)
    @test haskey(dfwres_focei, :iwres)
    @test haskey(dfwres_focei, :SEX)
    @test haskey(dfwres_focei, :WT)
    @test haskey(dfwres_focei_no_covar, :wres)
    @test haskey(dfwres_focei_no_covar, :iwres)
    @test !(haskey(dfwres_focei_no_covar, :SEX))
    @test !(haskey(dfwres_focei_no_covar, :WT))
  end

end

@testset "run2.mod FO without interaction, ODE solver, diagonal omega and additive error" begin
  #Note: run2 requires a block diagonal for the omega
  #$OMEGA  BLOCK(3)
  # 5.55  ;       KA__
  # 0.00524 0.00024  ;   COV KA~K
  # -0.128 0.00911 0.515  ;         K_
  theopmodel_solver_fo = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PSDDomain(3)
      σ_add ∈ RealDomain(lower=0.001, init=0.388)
      #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂+ η[2]
      CL = θ₃*WT + η[3]
      V  = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
      cp   = Central/V
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - (CL/V)*Central
    end

    @derived begin
      dv ~ @. Normal(conc,sqrt(σ_add))
    end
  end

  param = (
    θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
    Ω  = diagm(0 => [5.55, 0.0024, 0.515]), # update to block diagonal
    σ_add = 0.388
    # σ_prop = 0.3
       )

  @test deviance(theopmodel_solver_fo, theopp, param, PuMaS.FO(),reltol=1e-6,abstol=1e-8) ≈ 137.16573310096661

  fo_estimated_params = (θ₁ = 4.20241E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                         θ₂ = 7.25283E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                         θ₃ = 3.57499E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                         θ₄ = 2.12401E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

                         Ω = [1.81195E+01 -1.12474E-02 -3.05266E-02
                             -1.12474E-02  2.25098E-04  1.04586E-02
                             -3.05266E-02  1.04586E-02  5.99850E-01],
                         σ_add = 2.66533E-01)

  fo_stderr           = (θ₁ = 4.47E-01,
                         θ₂ = 6.81E-03,
                         θ₃ = 4.93E-03,
                         θ₄ = 3.52E-01,
                         Ω = [9.04E+00 1.92E-02 1.25E+00
                              1.92E-02 7.86E-05 4.57E-03
                              1.25E+00 4.57E-03 3.16E-01],
                         σ_add = 5.81E-02)
                         # Elapsed estimation time in seconds:     0.45
                         # Elapsed covariance time in seconds:     0.18

  o = fit(theopmodel_solver_fo, theopp, param, PuMaS.FO())

  o_estimates = o.param
  o_stderror  = stderror(o)

  o_infer = infer(o)

  # Verify that show runs
  io_buffer = IOBuffer()
  show(io_buffer, o)
  show(io_buffer, o_infer)

  @test deviance(o) ≈ 71.979975297638589 rtol=1e-6

  @testset "test estimate of $k" for k in keys(o_estimates)
    @test _extract(getfield(o_estimates, k)) ≈ _extract(getfield(fo_estimated_params, k)) rtol=1e-3
  end

  @testset "test stderror of $k" for k in keys(o_estimates)
    @test _extract(getfield(o_stderror, k))  ≈ _extract(getfield(fo_stderr, k))           rtol=2e-2
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test o.vvrandeffs[i] == zeros(3)
  end

  # Test that the types work on both stiff and non-stiff solver methods
  o = fit(theopmodel_solver_fo, theopp, param, PuMaS.FO(), alg=Tsit5())
  o = fit(theopmodel_solver_fo, theopp, param, PuMaS.FO(), alg=Rosenbrock23())
end

@testset "run3.mod FOCE without interaction, diagonal omega and additive error" begin
  theopmodel_foce = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PDiagDomain(2)
      σ_add ∈ RealDomain(lower=0.001, init=0.388)
      #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      V  = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics OneCompartmentModel

    @derived begin
      dv ~ @. Normal(conc,sqrt(σ_add))
    end
  end

  param = (θ₁ = 2.77,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = Diagonal([5.55, 0.515]),
        σ_add = 0.388
        #σ_prop = 0.3
       )

  @test deviance(theopmodel_foce, theopp, param, PuMaS.FOCE()) ≈ 138.90111320972699 rtol=1e-6

  foce_estimated_params = (
    θ₁ = 1.67977E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.49011E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.93898E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.10668E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω  = Diagonal([1.62087E+00, 2.26449E-01]),
    σ_add = 5.14069E-01)

  foce_stderr = (
    θ₁ = 1.84E-01,
    θ₂ = 5.43E-03,
    θ₃ = 3.42E-03,
    θ₄ = 9.56E-01,

    Ω  = Diagonal([1.61E+00, 7.70E-02]),
    σ_add = 1.34E-01)

  foce_ebes = [-2.66223E-01 -9.45749E-01
                4.53194E-01  3.03170E-02
                6.31423E-01  7.48592E-02
               -4.72536E-01 -1.80373E-01
               -1.75904E-01  1.64858E-01
               -4.31006E-01  4.74851E-01
               -1.33960E+00  4.29688E-01
               -6.61193E-01  3.15429E-01
                2.83927E+00 -6.59448E-01
               -1.46862E+00 -2.44162E-01
                1.47205E+00  6.97115E-01
               -1.12971E+00 -1.24210E-01]

  foce_ebes_cov = [2.96349E-02  6.43970E-03 6.43970E-03 5.95626E-03
                   1.13840E-01  1.97067E-02 1.97067E-02 1.53632E-02
                   1.37480E-01  2.07479E-02 2.07479E-02 1.42631E-02
                   2.91965E-02  9.78836E-03 9.78836E-03 1.35007E-02
                   3.48071E-02  7.73824E-03 7.73824E-03 7.30707E-03
                   5.87655E-02  2.22284E-02 2.22284E-02 3.92463E-02
                   1.58693E-02  9.82976E-03 9.82976E-03 2.39503E-02
                   5.74869E-02  1.64986E-02 1.64986E-02 2.20832E-02
                   8.51151E-01  3.17941E-02 3.17941E-02 1.25183E-02
                   4.95454E-03  3.20247E-03 3.20247E-03 6.51142E-03
                   4.08289E-01  3.73648E-02 3.73648E-02 2.03508E-02
                   1.46012E-02  5.21247E-03 5.21247E-03 7.67609E-03]

  # Elapsed estimation time in seconds:     0.27
  # Elapsed covariance time in seconds:     0.19

  o = fit(theopmodel_foce, theopp, param, PuMaS.FOCE())

  o_estimates = o.param
  o_stderror  = stderror(o)

  @test deviance(o) ≈ 121.89849119366599

  @testset "test estimate of $k" for k in keys(o_estimates)
    @test _extract(getfield(o_estimates, k)) ≈ _extract(getfield(foce_estimated_params, k)) rtol=1e-3
  end

  @testset "test stderror of $k" for k in keys(o_estimates)
    @test _extract(getfield(o_stderror, k))  ≈ _extract(getfield(foce_stderr, k))           rtol=1e-2
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test o.vvrandeffs[i] ≈ foce_ebes[i,:] rtol=1e-3
  end

  ebe_cov = PuMaS.empirical_bayes_dist(o)
  @testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test ebe_cov[i].η.Σ.mat[:] ≈ foce_ebes_cov[i,:] atol=1e-3
  end
end

@testset "run4.mod FOCEI, diagonal omega and additive + proportional error" begin
  theopmodel_focei = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PDiagDomain(2)
      σ_add ∈ RealDomain(lower=0.0001, init=0.388)
      σ_prop ∈ RealDomain(lower=0.0001, init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      V  = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics OneCompartmentModel

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2*σ_prop+σ_add))
    end
  end

  param = (θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = Diagonal([5.55, 0.515]),
        σ_add = 0.388,
        σ_prop = 0.3
       )

  @test deviance(theopmodel_focei, theopp, param, PuMaS.FOCEI()) ≈ 287.08854688950419 rtol=1e-6

  focei_estimated_params = (
    θ₁ = 1.58896E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.52144E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.97132E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.03889E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω = Diagonal([1.49637E+00, 2.62862E-01]),
    σ_add = 2.09834E-01,
    σ_prop = 1.13479E-02
  )

  focei_stderr = (
    θ₁ = 2.11E-01,
    θ₂ = 4.98E-03,
    θ₃ = 3.53E-03,
    θ₄ = 8.81E-01,

    Ω = Diagonal([1.35E+00, 8.06E-02]),
    σ_add = 2.64E-01,
    σ_prop = 1.35E-02)

  focei_ebes = [-3.85812E-01 -1.06806E+00
                 5.27060E-01  5.92269E-02
                 6.72391E-01  6.98525E-02
                -4.69064E-01 -1.90285E-01
                -2.15212E-01  1.46830E-01
                -4.06899E-01  5.11601E-01
                -1.30578E+00  4.62948E-01
                -5.67728E-01  3.40907E-01
                 2.62534E+00 -6.75318E-01
                -1.39161E+00 -2.55351E-01
                 1.46302E+00  7.37263E-01
                -1.12603E+00 -8.76924E-02]

  focei_ebes_cov = [2.48510E-02  7.25437E-03  7.25437E-03 9.35902E-03
                    1.15339E-01  2.18488E-02  2.18488E-02 2.01031E-02
                    1.38914E-01  2.24503E-02  2.24503E-02 1.85587E-02
                    2.46574E-02  1.03515E-02  1.03515E-02 1.78935E-02
                    3.40783E-02  9.74492E-03  9.74492E-03 1.18953E-02
                    4.00166E-02  1.82229E-02  1.82229E-02 3.92109E-02
                    1.18951E-02  8.95307E-03  8.95307E-03 2.70531E-02
                    5.26418E-02  1.67790E-02  1.67790E-02 2.53672E-02
                    8.16838E-01  3.06465E-02  3.06465E-02 1.54578E-02
                    5.91844E-03  4.37397E-03  4.37397E-03 1.05092E-02
                    3.96048E-01  3.79023E-02  3.79023E-02 2.46253E-02
                    1.41688E-02  6.63702E-03  6.63702E-03 1.28516E-02]

  # Elapsed estimation time in seconds:     0.30
  # Elapsed covariance time in seconds:     0.32

  o = fit(theopmodel_focei, theopp, param, PuMaS.FOCEI())

  o_estimates = o.param
  o_stderror  = stderror(o)

  o_infer = infer(o)

  # Verify that show runs
  io_buffer = IOBuffer()
  show(io_buffer, o)
  show(io_buffer, o_infer)

  @test deviance(o) ≈ 115.40505379554628 rtol=1e-7
  @testset "test estimate of $k" for k in keys(o_estimates)
    @test _extract(getfield(o_estimates, k)) ≈ _extract(getfield(focei_estimated_params, k)) rtol=1e-3
  end

  @testset "test stderror of $k" for k in keys(o_estimates)
    @test _extract(getfield(o_stderror, k))  ≈ _extract(getfield(focei_stderr, k))           rtol=1e-2
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test o.vvrandeffs[i] ≈ focei_ebes[i,:] rtol=1e-3
  end

  ebe_cov = PuMaS.empirical_bayes_dist(o)
  @testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test ebe_cov[i].η.Σ.mat[:] ≈ focei_ebes_cov[i,:] atol=1e-3
  end

  PuMaS.npde(   theopmodel_focei, theopp[1], param,
      (η=empirical_bayes(theopmodel_focei, theopp[1], param, PuMaS.FOCEI()),), 1000)
  PuMaS.epred(  theopmodel_focei, theopp[1], param,
      (η=empirical_bayes(theopmodel_focei, theopp[1], param, PuMaS.FOCEI()),), 1000)
  PuMaS.cpred(  theopmodel_focei, theopp[1], param)
  PuMaS.cpredi( theopmodel_focei, theopp[1], param)
  PuMaS.pred(   theopmodel_focei, theopp[1], param)
  PuMaS.wres(   theopmodel_focei, theopp[1], param)
  PuMaS.cwres(  theopmodel_focei, theopp[1], param)
  PuMaS.cwresi( theopmodel_focei, theopp[1], param)
  PuMaS.iwres(  theopmodel_focei, theopp[1], param)
  PuMaS.icwres( theopmodel_focei, theopp[1], param)
  PuMaS.icwresi(theopmodel_focei, theopp[1], param)
  PuMaS.eiwres( theopmodel_focei, theopp[1], param, 1000)
  PuMaS.cwres(  theopmodel_focei, theopp[1], param)
end

@testset "run4_foce.mod FOCE, diagonal omega and additive + proportional error" begin
  theopmodel_foce = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PDiagDomain(2)
      σ_add ∈ RealDomain(lower=0.0001, init=0.388)
      σ_prop ∈ RealDomain(lower=0.0001, init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      V  = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics OneCompartmentModel

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2*σ_prop+σ_add))
    end
  end

  param = (θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
             θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
             θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
             θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

             Ω = Diagonal([5.55, 0.515]),
             σ_add = 0.388,
             σ_prop = 0.3
            )

  foce_estimated_params = (
    θ₁ = 3.9553E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.2427E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.9067E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 3.2303E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω = Diagonal([6.1240E+00, 2.5828E-01]),
    σ_add = 1.1300E-01,
    σ_prop = 9.9131E-03
  )

  foce_ebes = [-2.69137E+00 -1.04910E+00
               -1.88515E+00  7.40755E-02
               -1.49661E+00  5.53721E-02
               -2.76850E+00 -1.92248E-01
               -2.51714E+00  1.60970E-01
               -2.68118E+00  5.20821E-01
               -2.44728E+00  4.57302E-01
               -1.72057E+00  3.20183E-01
                3.67973E+00 -6.68444E-01
               -2.59685E+00 -2.81286E-01
                8.06659E-01  7.37918E-01
               -2.25546E+00 -8.58804E-02]

  foce_ebes_cov = [2.15812E-02  3.05777E-03  3.05777E-03  3.85196E-03
                   1.29718E-01  1.70524E-02  1.70524E-02  1.59179E-02
                   1.92297E-01  1.99138E-02  1.99138E-02  1.41525E-02
                   3.48411E-02  6.72686E-03  6.72686E-03  1.16407E-02
                   6.10760E-02  7.92937E-03  7.92937E-03  1.00273E-02
                   6.72710E-02  1.55304E-02  1.55304E-02  3.32063E-02
                   2.24280E-02  6.89943E-03  6.89943E-03  2.28644E-02
                   7.18688E-02  1.36676E-02  1.36676E-02  2.10787E-02
                   2.25386E+00  4.60507E-02  4.60507E-02  8.15125E-03
                   7.32259E-03  2.11950E-03  2.11950E-03  5.87831E-03
                   7.04045E-01  5.72781E-02  5.72781E-02  2.61573E-02
                   2.21674E-02  4.24659E-03  4.24659E-03  8.75169E-03]

  # Elapsed estimation time in seconds:     0.34
  # Elapsed covariance time in seconds:     0.31

  o = fit(theopmodel_foce, theopp, param, PuMaS.FOCE())

  o_estimates = o.param

  @test deviance(o) ≈ 102.871158475488 rtol=1e-5

  @testset "test parameter $k" for k in keys(o_estimates)
    @test _extract(getfield(o_estimates, k)) ≈ _extract(getfield(foce_estimated_params, k)) rtol=1e-3
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test o.vvrandeffs[i] ≈ foce_ebes[i,:] rtol=1e-3
  end

  ebe_cov = PuMaS.empirical_bayes_dist(o)
  @testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test ebe_cov[i].η.Σ.mat[:] ≈ foce_ebes_cov[i,:] atol=1e-3
  end
end

@testset "run5.mod Laplace without interaction, diagonal omega and additive error" begin

  theopmodel_laplace = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PDiagDomain(2)
      σ_add ∈ RealDomain(lower=0.0001, init=0.388)
      #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      V  = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
        conc = Central / SC
    end

    @dynamics OneCompartmentModel

    @derived begin
        dv ~ @. Normal(conc, sqrt(σ_add))
    end
  end

  param = (θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = Diagonal([5.55, 0.515 ]),
        σ_add = 0.388
        #σ_prop = 0.3
       )

  nonmem_ebes_initial = [[-1.29964E+00, -8.32445E-01],
                         [-5.02283E-01,  1.04703E-01],
                         [-2.43001E-01,  1.51238E-01],
                         [-1.51174E+00, -1.02357E-01],
                         [-1.19255E+00,  2.15326E-01],
                         [-1.43398E+00,  5.89950E-01],
                         [-6.90249E-01,  5.11417E-01],
                         [ 1.15977E-02,  3.87675E-01],
                         [ 4.83842E+00, -5.54201E-01],
                         [-8.29081E-01, -1.63510E-01],
                         [ 2.63035E+00,  7.77899E-01],
                         [-4.82787E-01, -5.32906E-02]]

  @testset "Empirical Bayes estimates" begin
    for (i,η) in enumerate(nonmem_ebes_initial)
      @test empirical_bayes(theopmodel_laplace, theopp[i], param, PuMaS.Laplace()) ≈ η rtol=1e-4
    end

    @test deviance(theopmodel_laplace, theopp, param, PuMaS.Laplace()) ≈ 141.296 atol=1e-3
  end

  laplace_estimated_params = (
    θ₁ = 1.68975E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.54637E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.95757E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.11952E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω = Diagonal([1.596, 2.27638e-01]),
    σ_add = 5.14457E-01
  )

  laplace_stderr = (
    θ₁ = 1.87E-01,
    θ₂ = 5.45E-03,
    θ₃ = 3.42E-03,
    θ₄ = 9.69E-01,

    Ω = Diagonal([1.60E+00, 7.76E-02]),
    σ_add = 1.35E-01)

  laplace_ebes =[-2.81817E-01 -9.51075E-01
                  4.36123E-01  2.95020E-02
                  6.11776E-01  7.42657E-02
                 -4.86815E-01 -1.82637E-01
                 -1.91050E-01  1.64544E-01
                 -4.45223E-01  4.75049E-01
                 -1.35543E+00  4.29076E-01
                 -6.79064E-01  3.15307E-01
                  2.80934E+00 -6.62060E-01
                 -1.48446E+00 -2.47635E-01
                  1.44666E+00  6.99712E-01
                 -1.14629E+00 -1.26537E-01]

  laplace_ebes_cov = [2.93845E-02  6.58620E-03  6.58620E-03  6.15918E-03
                      8.77655E-02  1.52441E-02  1.52441E-02  1.46969E-02
                      1.34060E-01  2.02319E-02  2.02319E-02  1.42611E-02
                      2.50439E-02  8.56810E-03  8.56810E-03  1.33066E-02
                      2.80689E-02  6.29305E-03  6.29305E-03  7.02909E-03
                      4.82183E-02  1.79056E-02  1.79056E-02  3.66785E-02
                      1.42044E-02  8.76942E-03  8.76942E-03  2.30211E-02
                      5.16763E-02  1.48793E-02  1.48793E-02  2.15167E-02
                      5.14173E-01  1.61073E-02  1.61073E-02  1.22365E-02
                      5.22194E-03  3.44274E-03  3.44274E-03  6.76806E-03
                      3.23248E-01  2.75247E-02  2.75247E-02  1.88999E-02
                      1.23520E-02  4.49397E-03  4.49397E-03  7.51682E-03]

  # Elapsed estimation time in seconds:     0.23
  # Elapsed covariance time in seconds:     0.17

  @test deviance(theopmodel_laplace, theopp, laplace_estimated_params, Laplace()) ≈ 123.76439574418291 atol=1e-3

  @testset "Test optimization" begin
    o = fit(theopmodel_laplace, theopp, param, PuMaS.Laplace())

    o_estimates = o.param
    o_stderror  = stderror(o)

    o_infer = infer(o)

    # Verify that show runs
    io_buffer = IOBuffer()
    show(io_buffer, o)
    show(io_buffer, o_infer)

    @test deviance(o) ≈ 123.76439574418291 rtol=1e-5

    @testset "test estimate of $k" for k in keys(o_estimates)
      @test _extract(getfield(o_estimates, k)) ≈ _extract(getfield(laplace_estimated_params, k)) rtol=2e-3
    end

    @testset "test stderror of $k" for k in keys(o_estimates)
      @test _extract(getfield(o_stderror, k))  ≈ _extract(getfield(laplace_stderr, k))           rtol=1e-2
    end

    @testset "test stored empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
      @test o.vvrandeffs[i] ≈ laplace_ebes[i,:] rtol=3e-3
    end

    ebe_cov = PuMaS.empirical_bayes_dist(o)
    @testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
        @test ebe_cov[i].η.Σ.mat[:] ≈ laplace_ebes_cov[i,:] atol=1e-3
    end
  end
end

@testset "run6.mod LaplaceI, diagonal omega and additive + proportional error" begin

  theopmodel_laplacei = @model begin
    @param begin
      # θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      # θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      # θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      # θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      # Use VectorDomain although RealDomain is preferred to make sure that VectorDomain is tested
      θ ∈ VectorDomain(4, lower=[0.1 , 0.0008, 0.004 , 0.1],
                          init =[2.77, 0.0781, 0.0363, 1.5],
                          upper=[5.0 , 0.5   , 0.9   , 5.0])
      Ω ∈ PDiagDomain(2)
      σ_add ∈ RealDomain(lower=0.0001, init=0.388)
      σ_prop ∈ RealDomain(lower=0.0001, init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ[1] + η[1] : θ[4] + η[1]
      K  = θ[2]
      CL = θ[3]*WT + η[2]
      V  = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics OneCompartmentModel

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2*σ_prop+σ_add))
    end
  end

  param = (θ = [2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                1.5],   #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = Diagonal([5.55,0.515]),
        σ_add = 0.388,
        σ_prop = 0.3
       )

  @test deviance(theopmodel_laplacei, theopp, param, PuMaS.LaplaceI()) ≈ 288.30901928585990 rtol=1e-6

  laplacei_estimated_params = (
    θ = [1.60941E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
         8.55663E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
         3.97472E-02,  #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
         2.05830E+00], #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω = Diagonal([1.48117E+00, 2.67215E-01]),
    σ_add = 1.88050E-01,
    σ_prop = 1.25319E-02
  )

  laplacei_stderr = (
    θ = [2.09E-01,
         4.48E-03,
         3.29E-03,
         9.05E-01],

    Ω = Diagonal([1.35E+00, 8.95E-02]),
    σ_add = 3.01E-01,
    σ_prop = 1.70E-02)

  laplacei_ebes = [-4.28671E-01 -1.07834E+00
                    5.07444E-01  6.93096E-02
                    6.43525E-01  7.59602E-02
                   -4.98953E-01 -1.84868E-01
                   -2.46292E-01  1.51559E-01
                   -4.38459E-01  5.21590E-01
                   -1.33160E+00  4.71199E-01
                   -5.83976E-01  3.49324E-01
                    2.57215E+00 -6.69523E-01
                   -1.41344E+00 -2.53978E-01
                    1.43116E+00  7.48520E-01
                   -1.15300E+00 -7.93525E-02]

  laplacei_ebes_cov = [2.50922E-02  8.73478E-03  8.73478E-03  1.09246E-02
                       1.05367E-01  2.07642E-02  2.07642E-02  2.00787E-02
                       1.32364E-01  2.21959E-02  2.21959E-02  1.90778E-02
                       2.27392E-02  1.05318E-02  1.05318E-02  1.84061E-02
                       3.29183E-02  1.03562E-02  1.03562E-02  1.25563E-02
                       3.37488E-02  1.74040E-02  1.74040E-02  3.91317E-02
                       1.04365E-02  8.73379E-03  8.73379E-03  2.72465E-02
                       5.56885E-02  1.72519E-02  1.72519E-02  2.57910E-02
                       4.58448E-01  2.47681E-02  2.47681E-02  1.58825E-02
                       6.12637E-03  4.60375E-03  4.60375E-03  1.11188E-02
                       3.11433E-01  3.31735E-02  3.31735E-02  2.47868E-02
                       1.23976E-02  6.46725E-03  6.46725E-03  1.29075E-02]

  # Elapsed estimation time in seconds:     0.30
  # Elapsed covariance time in seconds:     0.32

  @testset "Test optimization" begin
    o = fit(theopmodel_laplacei, theopp, param, PuMaS.LaplaceI())

    o_estimates = o.param
    o_stderror  = stderror(o)

    @test deviance(o) ≈ 116.97275684239327 rtol=1e-5

    @testset "test estimate of $k" for k in keys(o_estimates)
      @test _extract(getfield(o_estimates, k)) ≈ _extract(getfield(laplacei_estimated_params, k)) rtol=1e-3
    end

    @testset "test stderror of $k" for k in keys(o_estimates)
      @test _extract(getfield(o_stderror, k))  ≈ _extract(getfield(laplacei_stderr, k))           rtol=4e-2
    end

    @testset "test stored empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
      @test o.vvrandeffs[i] ≈ laplacei_ebes[i,:] rtol=1e-3
    end

    ebe_cov = PuMaS.empirical_bayes_dist(o)
    @testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
      @test ebe_cov[i].η.Σ.mat[:] ≈ laplacei_ebes_cov[i,:] rtol=1e-3
    end
  end
end
end
