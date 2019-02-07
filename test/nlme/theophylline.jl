using Test
using PuMaS, LinearAlgebra, Optim
# FIXME! Remove this dependency once we have a fit function and therefore don't need to call the transforms explicitly
using TransformVariables
using PuMaS: totransform

# FIXME! Find a nicer way to handle this
_extract(A::PDMat) = A.mat
_extract(A::PDiagMat) = A.diag
_extract(A) = A

@testset "Theophylline model" begin

# Control verbosity of solver output
verbose = false

theopp = process_nmtran(example_nmtran_data("event_data/THEOPP"),[:SEX,:WT])

@testset "run2.mod FO without interaction, diagonal omega and additive error, \$COV = sandwich matrix" begin
  #Note: run2 requires a block diagonal for the omega
  #$OMEGA  BLOCK(3)
  # 5.55  ;       KA__
  # 0.00524 0.00024  ;   COV KA~K
  # -0.128 0.00911 0.515  ;         K_
  theopmodel_fo = @model begin
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

  x0 = (
    θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
    Ω = PDMat(diagm(0 => [5.55, 0.0024, 0.515])), # update to block diagonal
    σ_add = 0.388
    # σ_prop = 0.3
       )

  @test PuMaS.marginal_nll_nonmem(theopmodel_fo, theopp, x0, PuMaS.FO()) ≈ 137.16573310096661

  fo_estimated_params = (θ₁ = 4.20241E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                         θ₂ = 7.25283E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                         θ₃ = 3.57499E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                         θ₄ = 2.12401E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

                         Ω = [1.81195E+01 -1.12474E-02 -3.05266E-02
                             -1.12474E-02  2.25098E-04  1.04586E-02
                             -3.05266E-02  1.04586E-02  5.99850E-01],
                         σ_add = 2.66533E-01)
                         # Elapsed estimation time in seconds:     0.04
                         # Elapsed covariance time in seconds:     0.02

  o = optimize(t -> PuMaS.marginal_nll_nonmem(theopmodel_fo, # The marginal likelihood is the objective
                                              theopp,
                                              TransformVariables.transform(totransform(theopmodel_fo.param), t),
                                              PuMaS.FO()),
               TransformVariables.inverse(totransform(theopmodel_fo.param), x0), # The initial values
               BFGS(),                                                           # The optimization method
               Optim.Options(show_trace=verbose,                                 # Print progress
                             g_tol=1e-5))                                        # Adjust convergence tolerance

  x_optim = TransformVariables.transform(totransform(theopmodel_fo.param), o.minimizer)

  @test o.f_converged
  @test o.minimum ≈ 71.979975297638589
  @testset "test parameter $k" for k in keys(x_optim)
    @test _extract(getfield(x_optim, k)) ≈ getfield(fo_estimated_params, k) rtol=1e-3
  end
end

@testset "run2.mod FO without interaction, ODE solver, diagonal omega and additive error, \$COV = sandwich matrix" begin
  #Note: run2 requires a block diagonal for the omega
  #$OMEGA  BLOCK(3)
  # 5.55  ;       KA__
  # 0.00524 0.00024  ;   COV KA~K
  # -0.128 0.00911 0.515  ;         K_
  theopmodel_fo = @model begin
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

    @dynamics begin
        cp       =  Central/V
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - (CL/V)*Central
    end

    @derived begin
      dv ~ @. Normal(conc,sqrt(σ_add))
    end
  end

  x0 = (
    θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
    Ω = PDMat(diagm(0 => [5.55, 0.0024, 0.515])), # update to block diagonal
    σ_add = 0.388
    # σ_prop = 0.3
       )

  @test PuMaS.marginal_nll_nonmem(theopmodel_fo, theopp, x0, PuMaS.FO()) ≈ 137.16573310096661

  #=
  # Would stall Travis
  fo_estimated_params = (θ₁ = 4.20241E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                         θ₂ = 7.25283E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                         θ₃ = 3.57499E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                         θ₄ = 2.12401E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

                         Ω = [1.81195E+01 -1.12474E-02 -3.05266E-02
                             -1.12474E-02  2.25098E-04  1.04586E-02
                             -3.05266E-02  1.04586E-02  5.99850E-01],
                         σ_add = 2.66533E-01)
                         # Elapsed estimation time in seconds:     0.04
                         # Elapsed covariance time in seconds:     0.02

  o = optimize(t -> PuMaS.marginal_nll_nonmem(theopmodel_fo, # The marginal likelihood is the objective
                                              theopp,
                                              TransformVariables.transform(totransform(theopmodel_fo.param), t),
                                              PuMaS.FO()),
               TransformVariables.inverse(totransform(theopmodel_fo.param), x0), # The initial values
               BFGS(),                                                           # The optimization method
               Optim.Options(show_trace=verbose,                                 # Print progress
                             g_tol=1e-5))                                        # Adjust convergence tolerance

  x_optim = TransformVariables.transform(totransform(theopmodel_fo.param), o.minimizer)

  @test o.f_converged
  @test o.minimum ≈ 71.979975297638589
  @testset "test parameter $k" for k in keys(x_optim)
    @test _extract(getfield(x_optim, k)) ≈ getfield(fo_estimated_params, k) rtol=1e-3
  end
  =#
end

@testset "run3.mod FOCE without interaction, diagonal omega and additive error, \$COV = sandwich matrix" begin
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

  x0 = (θ₁ = 2.77,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = PDiagMat([5.55, 0.515]),
        σ_add = 0.388
        #σ_prop = 0.3
       )

  @test PuMaS.marginal_nll_nonmem(theopmodel_foce, theopp, x0, PuMaS.FOCE()) ≈ 138.90111320972699

  foce_estimated_params = (
    θ₁ = 1.67977E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.49011E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.93898E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.10668E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω  = PDiagMat([1.62087E+00, 2.26449E-01]),
    σ_add = 5.14069E-01)
    # Elapsed estimation time in seconds:     0.27
    # Elapsed covariance time in seconds:     0.19

  o = optimize(t -> PuMaS.marginal_nll_nonmem(theopmodel_foce, # The marginal likelihood is the objective
                                              theopp,
                                              TransformVariables.transform(totransform(theopmodel_foce.param), t),
                                              PuMaS.FOCE()),
               TransformVariables.inverse(totransform(theopmodel_foce.param), x0), # The initial values
               BFGS(),                                                           # The optimization method
               Optim.Options(show_trace=verbose,                                 # Print progress
                             g_tol=1e-5))

  x_optim = TransformVariables.transform(totransform(theopmodel_foce.param), o.minimizer)

  @test_broken o.f_converged
  @test o.minimum ≈ 121.89849118143030
  @testset "test parameter $k" for k in keys(x_optim)
    @test _extract(getfield(x_optim, k)) ≈ _extract(getfield(foce_estimated_params, k)) rtol=1e-3
  end
end

# FIXME! Temporary workaround to avoid that NaN from the ODE solve propagate
_nonan(x) = isnan(x) ? floatmin() : x

@testset "run4.mod FOCE with interaction, diagonal omega and additive + proportional error, \$COV = sandwich matrix" begin
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
      dv ~ @. Normal(conc, _nonan(sqrt(conc^2*σ_prop+σ_add)))
    end
  end

  x0 = (θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = PDiagMat([5.55, 0.515]),
        σ_add = 0.388,
        σ_prop = 0.3
       )

  @test PuMaS.marginal_nll_nonmem(theopmodel_focei, theopp, x0, PuMaS.FOCEI()) ≈ 287.08854688950419

  focei_estimated_params = (
    θ₁ = 1.58896E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.52144E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.97132E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.03889E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω = PDiagMat([1.49637E+00, 2.62862E-01]),
    σ_add = 2.09834E-01,
    σ_prop = 1.13479E-02
  )
  # Elapsed estimation time in seconds:     0.30
  # Elapsed covariance time in seconds:     0.32

  o = optimize(t -> PuMaS.marginal_nll_nonmem(theopmodel_focei, # The marginal likelihood is the objective
                                              theopp,
                                              TransformVariables.transform(totransform(theopmodel_focei.param), t),
                                              PuMaS.FOCEI()),
               TransformVariables.inverse(totransform(theopmodel_focei.param), x0), # The initial values
               BFGS(),                                                           # The optimization method
               Optim.Options(show_trace=verbose,                                 # Print progress
                             g_tol=1e-5))

  x_optim = TransformVariables.transform(totransform(theopmodel_focei.param), o.minimizer)

  @test_broken o.f_converged
  @test o.minimum ≈ 115.40505381367598 rtol=1e-5
  @testset "test parameter $k" for k in keys(x_optim)
    @test _extract(getfield(x_optim, k)) ≈ _extract(getfield(focei_estimated_params, k)) rtol=1e-3
  end
end

@testset "run5.mod Laplace without interaction, diagonal omega and additive error, \$COV = sandwich matrix" begin

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

  x0 = (θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = PDiagMat([5.55, 0.515 ]),
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
      @test PuMaS.rfx_estimate(theopmodel_laplace, theopp[i], x0, PuMaS.Laplace()) ≈ η rtol=1e-4
    end

    @test PuMaS.marginal_nll_nonmem(theopmodel_laplace, theopp, x0, PuMaS.Laplace()) ≈ 141.296 atol=1e-3
  end

  laplace_estimated_params = (
    θ₁ = 1.68975E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.54637E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.95757E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.11952E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω = PDiagMat([1.596, 2.27638e-01]),
    σ_add = 5.14457E-01
  )
  # Elapsed estimation time in seconds:     0.23
  # Elapsed covariance time in seconds:     0.17

  # from run5.phi
  nonmem_ebes_estimated = [[-2.81817E-01, -9.51075E-01],
                           [ 4.36123E-01,  2.95020E-02],
                           [ 6.11776E-01,  7.42657E-02],
                           [-4.86815E-01, -1.82637E-01],
                           [-1.91050E-01,  1.64544E-01],
                           [-4.45223E-01,  4.75049E-01],
                           [-1.35543E+00,  4.29076E-01],
                           [-6.79064E-01,  3.15307E-01],
                           [ 2.80934E+00, -6.62060E-01],
                           [-1.48446E+00, -2.47635E-01],
                           [ 1.44666E+00,  6.99712E-01],
                           [-1.14629E+00, -1.26537E-01]]

  @testset "Laplace fitted" begin
    for (i,η) in enumerate(nonmem_ebes_estimated)
      @test PuMaS.rfx_estimate(theopmodel_laplace, theopp[i], laplace_estimated_params, Laplace()) ≈ η rtol=1e-4
    end

    @test PuMaS.marginal_nll_nonmem(theopmodel_laplace, theopp, laplace_estimated_params, Laplace()) ≈ 123.76439574418291 atol=1e-3
  end

  @testset "Test optimization" begin
    o = optimize(t -> PuMaS.marginal_nll_nonmem(theopmodel_laplace, # The marginal likelihood is the objective
                                                theopp,
                                                TransformVariables.transform(totransform(theopmodel_laplace.param), t),
                                                PuMaS.Laplace()),
                 TransformVariables.inverse(totransform(theopmodel_laplace.param), x0), # The initial values
                 BFGS(),                                                           # The optimization method
                 Optim.Options(show_trace=verbose,                                 # Print progress
                               g_tol=1e-5))

    x_optim = TransformVariables.transform(totransform(theopmodel_laplace.param), o.minimizer)

    @test_broken o.f_converged
    @test o.minimum ≈ 123.76439574418291 rtol=1e-5
    @testset "test parameter $k" for k in keys(x_optim)
      @test _extract(getfield(x_optim, k)) ≈ _extract(getfield(laplace_estimated_params, k)) rtol=1e-3
    end
  end

end

@testset "run6.mod Laplace with interaction, diagonal omega and additive + proportional error , \$COV = sandwich matrix" begin

  theopmodel_laplacei = @model begin
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
      dv ~ @. Normal(conc, _nonan(sqrt(conc^2*σ_prop+σ_add)))
    end
  end

  x0 = (θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = PDiagMat([5.55,0.515]),
        σ_add = 0.388,
        σ_prop = 0.3
       )

  @test PuMaS.marginal_nll_nonmem(theopmodel_laplacei, theopp, x0, PuMaS.LaplaceI()) ≈ 288.30901928585990

  laplacei_estimated_params = (
    θ₁ = 1.60941E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.55663E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.97472E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.05830E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω = PDiagMat([1.48117E+00, 2.67215E-01]),
    σ_add = 1.88050E-01,
    σ_prop = 1.25319E-02
  )
  # Elapsed estimation time in seconds:     0.30
  # Elapsed covariance time in seconds:     0.32

  @testset "Test optimization" begin
    o = optimize(t -> PuMaS.marginal_nll_nonmem(theopmodel_laplacei, # The marginal likelihood is the objective
                                                theopp,
                                                TransformVariables.transform(totransform(theopmodel_laplacei.param), t),
                                                PuMaS.LaplaceI()),
                 TransformVariables.inverse(totransform(theopmodel_laplacei.param), x0), # The initial values
                 BFGS(),                                                           # The optimization method
                 Optim.Options(show_trace=verbose,                                 # Print progress
                               g_tol=1e-5))

    x_optim = TransformVariables.transform(totransform(theopmodel_laplacei.param), o.minimizer)

    @test_broken o.f_converged
    @test o.minimum ≈ 116.97275684239327 rtol=1e-5
    @testset "test parameter $k" for k in keys(x_optim)
      @test _extract(getfield(x_optim, k)) ≈ _extract(getfield(laplacei_estimated_params, k)) rtol=1e-3
    end
  end
end
end
