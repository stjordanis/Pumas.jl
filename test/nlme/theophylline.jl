using Test
using PuMaS, LinearAlgebra, Optim

@testset "Theophylline model" begin

theopp = process_nmtran(example_nmtran_data("event_data/THEOPP"),[:SEX,:WT])

@testset "run2.mod FO without interaction, diagonal omega and additive error, \$COV = sandwich matrix" begin
  #Note: run2 requires a block diagonal for the omega
  #$OMEGA  BLOCK(3)
  # 5.55  ;       KA__
  # 0.00524 0.00024  ;   COV KA~K
  # -0.128 0.00911 0.515  ;         K_
  theopmodel_fo = @model begin
    @param begin
      θ ∈ VectorDomain(4,
                       lower=[0.1,0.0008,0.004,0.1],
                       init=[2.77,0.0781,0.0363,1.5],
                       upper =[5,0.5,0.9,5])
      Ω ∈ PSDDomain(3)
      σ_add ∈ RealDomain(init=0.388)
      #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ[1] + η[1] : θ[4] + η[1]
      K = θ[2]+ η[2]
      CL  = θ[3]*WT + η[3]
      V = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / V
    end

    @dynamics ImmediateAbsorptionModel

    @derived begin
      dv ~ @. Normal(conc,sqrt(σ_add))
      end
  end

  x0 = (θ = [2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
             0.0781,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
             0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
             1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
            ],
        Ω = PDMat(diagm(0 => [5.55, 0.0024, 0.515])), # update to block diagonal
        σ_add = 0.388
        #σ_prop = 0.3
       )

  fo_obj = 71.979975297638589
  fo_estimated_params = (θ = [4.20241E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                              7.25283E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                              3.57499E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                              2.12401E+00 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
                             ],
                         Ω_11  = 1.81195E+01,
                         Ω_21  = -1.12474E-02,
                         Ω_22  = 2.25098E-04,
                         Ω_31  = -3.05266E-02,
                         Ω_32  = 1.04586E-02,
                         Ω_33  = 5.99850E-01,
                         σ_add = 2.66533E-01)
                         # Elapsed estimation time in seconds:     0.04
                         # Elapsed covariance time in seconds:     0.02
end

@testset "run3.mod FOCE without interaction, diagonal omega and additive error, \$COV = sandwich matrix" begin
  theopmodel_foce = @model begin
    @param begin
      θ ∈ VectorDomain(4,
                       lower=[0.1,0.0008,0.004,0.1],
                       init=[2.77,0.0781,0.0363,1.5],
                       upper =[5,0.5,0.9,5])
      Ω ∈ PSDDomain(2)
      σ_add ∈ RealDomain(init=0.388)
      #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ[1] + η[1] : θ[4] + η[1]
      K  = θ[2]
      CL = θ[3]*WT + η[2]
      V  = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / V
    end

    @dynamics ImmediateAbsorptionModel

    @derived begin
      dv ~ @. Normal(conc,sqrt(σ_add))
    end
  end

  x0 = (θ = [2.77,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
             0.0781,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
             0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
             1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
            ],
        Ω = PDMat(diagm(0 => [5.55,0.515])),
        σ_add = 0.388
        #σ_prop = 0.3
       )

  foce_obj = 121.89849118143030
  foce_estimated_params = (
    θ = [1.67977E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
         8.49011E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
         3.93898E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
         2.10668E+00  #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
        ],
    Ω_11  = 1.62087E+00,
    Ω_21  = 0.00000E+00,
    Ω_22  = 2.26449E-01,
    σ_add = 5.14069E-01)
    # Elapsed estimation time in seconds:     0.27
    # Elapsed covariance time in seconds:     0.19
end

@testset "run4.mod FOCE with interaction, diagonal omega and additive + proportional error, \$COV = sandwich matrix" begin
  theopmodel_focei = @model begin
    @param begin
      θ ∈ VectorDomain(4,
                       lower=[0.1,0.0008,0.004,0.1],
                       init=[2.77,0.0781,0.0363,1.5],
                       upper =[5,0.5,0.9,5])
      Ω ∈ PSDDomain(2)
      σ_add ∈ RealDomain(init=0.388)
      σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ[1] + η[1] : θ[4] + η[1]
      K  = θ[2]
      CL = θ[3]*WT + η[2]
      V = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / V
    end

    @dynamics ImmediateAbsorptionModel

    @derived begin
      dv ~ @. Normal(conc,conc*sqrt(σ_prop)+(σ_add))
    end
  end

  x0 = (θ = [2.77,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
             0.0781,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
             0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
              1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
            ],
        Ω = PDMat(diagm(0 => [5.55,0.515])),
        σ_add = 0.388,
        σ_prop = 0.3
       )

  focei_obj = 115.40505381367598
  focei_estimated_params = (
    θ = [1.58896E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
         8.52144E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
         3.97132E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
         2.03889E+00 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
        ],
    Ω_11  = 1.49637E+00,
    Ω_21  = 0.00000E+00,
    Ω_22  = 2.62862E-01,
    σ_add = 2.09834E-01,
    σ_prop = 1.13479E-02
  )
  # Elapsed estimation time in seconds:     0.30
  # Elapsed covariance time in seconds:     0.32
end

@testset "run5.mod Laplace without interaction, diagonal omega and additive error, \$COV = sandwich matrix" begin

  theopmodel_laplace = @model begin
    @param begin
      θ ∈ VectorDomain(4,
                       lower=[0.1,0.0008,0.004,0.1],
                       init=[2.77,0.0781,0.0363,1.5],
                       upper =[5,0.5,0.9,5])
      Ω ∈ PSDDomain(2)
      σ_add ∈ RealDomain(init=0.388)
      #σ_prop ∈ RealDomain(init=0.3)
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
        dv ~ @. Normal(conc,sqrt(σ_add))
    end
  end

  x0 = (θ = [2.77,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
             0.0781,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
             0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
             1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
            ],
        Ω = PDMat(diagm(0 => [5.55, 0.515 ])),
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

    @test_broken PuMaS.marginal_nll_nonmem(theopmodel_laplace, theopp, x0, PuMaS.Laplace()) ≈ 141.296 atol=1e-3
  end

  laplace_estimated_params = (
    θ = [1.68975E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
         8.54637E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
         3.95757E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
         2.11952E+00 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
        ],
    Ω = PDMat([1.596 0.0; 0.0 2.27638e-01]),
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

    @test_broken PuMaS.marginal_nll_nonmem(theopmodel_laplace, theopp, laplace_estimated_params, Laplace()) ≈ 123.76439574418291 atol=1e-3
  end
end

@testset "run6.mod Laplace with interaction, diagonal omega and additive + proportional error , \$COV = sandwich matrix" begin

  theopmodel_laplacei = @model begin
    @param begin
      θ ∈ VectorDomain(4,
                       lower=[0.1,0.0008,0.004,0.1],
                       init=[2.77,0.0781,0.0363,1.5],
                       upper =[5,0.5,0.9,5])
      Ω ∈ PSDDomain(diagm(0 => [5.55,0.515]))
      σ_add ∈ RealDomain(init=0.388)
      σ_prop ∈ RealDomain(init=0.3)
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

    @dynamics ImmediateAbsorptionModel

    @derived begin
      # dv ~ @. Normal(conc,conc*sqrt(σ_prop)+(σ_add))
      dv ~ @. Normal(conc,conc*σ_prop + σ_add)
      # dv ~ @. Normal(conc,sqrt(conc^2*σ_prop+σ_add))
      # dv ~ @. Normal(conc,sqrt(conc^2+σ_add))
    end
  end

  x0 = (θ = [2.77,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
             0.0781,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
             0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
             1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
            ],
        Ω = PDMat(diagm(0 => [5.55,0.515])),
        σ_add = 0.388,
        σ_prop = 0.3
       )

  laplacei_obj = 116.97275684239327
  laplacei_estimated_params = (
    θ = [1.60941E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
         8.55663E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
         3.97472E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
         2.05830E+00 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
        ],
    Ω_11  = 1.48117E+00,
    Ω_21  = 0.00000E+00,
    Ω_22  = 2.67215E-01,
    σ_add = 1.88050E-01,
    σ_prop = 1.25319E-02
  )
  # Elapsed estimation time in seconds:     0.30
  # Elapsed covariance time in seconds:     0.32
end
end
