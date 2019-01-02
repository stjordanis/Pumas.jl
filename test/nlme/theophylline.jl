using Test
using PuMaS, LinearAlgebra, Optim

theopp = process_nmtran(example_nmtran_data("event_data/THEOPP"),[:SEX,:WT])

# Theophylline model

#run2.mod FO without interaction, diagonal omega and additive error, $COV = sandwich matrix
#Note: run2 requires a block diagonal for the omega
#$OMEGA  BLOCK(3)
# 5.55  ;       KA__
# 0.00524 0.00024  ;   COV KA~K
# -0.128 0.00911 0.515  ;         K_
theopmodel_fo = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=[0.1,0.0008,0.004,0.1], init=[2.77,0.0781,0.0363,1.5], upper =[5,0.5,0.9,5])
        Ω ∈ PSDDomain(3)
        σ_add ∈ RealDomain(init=0.388)
        #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka = SEX == 1 ? θ[1] + η[1] : θ[4] + η[1]
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


# method = fo
x0 = (θ = [2.7,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
           0.078,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
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
                         #  Elapsed covariance time in seconds:     0.02

#  run3.mod FOCE without interaction, diagonal omega and additive error, $COV = sandwich matrix
# method = foce
theopmodel_foce = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=[0.1,0.0008,0.004,0.1], init=[2.77,0.0781,0.0363,1.5], upper =[5,0.5,0.9,5])
        Ω ∈ PSDDomain(2)
        σ_add ∈ RealDomain(init=0.388)
        #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka = SEX == 1 ? θ[1] + η[1] : θ[4] + η[1]
        K = θ[2]
        CL  = θ[3]*WT + η[2]
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
x0 = (θ = [2.7,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
           0.078,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
           0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
           1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
           ],
           Ω = PDMat(diagm(0 => [5.55,0.515])),
           σ_add = 0.388
           #σ_prop = 0.3
           )

foce_obj = 121.89849118143030
foce_estimated_params = (θ = [1.67977E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                           8.49011E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                           3.93898E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                           2.10668E+00 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
                           ],
                           Ω_11  = 1.62087E+00,
                           Ω_21  = 0.00000E+00,
                           Ω_22  = 2.26449E-01,
                           σ_add = 5.14069E-01)
# Elapsed estimation time in seconds:     0.27
# Elapsed covariance time in seconds:     0.19

#  run4.mod FOCE with interaction, diagonal omega and additive + proportional error, $COV = sandwich matrix
# method = focei
theopmodel_focei = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=[0.1,0.0008,0.004,0.1], init=[2.77,0.0781,0.0363,1.5], upper =[5,0.5,0.9,5])
        Ω ∈ PSDDomain(2)
        σ_add ∈ RealDomain(init=0.388)
        σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka = SEX == 1 ? θ[1] + η[1] : θ[4] + η[1]
        K = θ[2]
        CL  = θ[3]*WT + η[2]
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
x0 = (θ = [2.7,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
           0.078,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
           0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
           1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
           ],
           Ω = PDMat(diagm(0 => [5.55,0.515])),
           σ_add = 0.388,
           σ_prop = 0.3
           )

focei_obj = 115.40505381367598
focei_estimated_params = (θ = [1.58896E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                           8.52144E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                           3.97132E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                           2.03889E+00 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
                           ],
                           Ω_11  = 1.49637E+00,
                           Ω_21  = 0.00000E+00,
                           Ω_22  = 2.62862E-01,
                           σ_add = 2.09834E-01,
                           σ_prop = 1.13479E-02)
# Elapsed estimation time in seconds:     0.30
#  Elapsed covariance time in seconds:     0.32

# run5.mod Laplace without interaction, diagonal omega and additive error, $COV = sandwich matrix
# method = laplace
theopmodel_laplace = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=[0.1,0.0008,0.004,0.1], init=[2.77,0.0781,0.0363,1.5], upper =[5,0.5,0.9,5])
        Ω ∈ PSDDomain(2)
        σ_add ∈ RealDomain(init=0.388)
        #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka = SEX == 1 ? θ[1] + η[1] : θ[4] + η[1]
        K = θ[2]
        CL  = θ[3]*WT + η[2]
        V = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
        conc = Central / V
    end

    @dynamics OneCompartmentModel

    @derived begin
        dv ~ @. Normal(conc,sqrt(σ_add))
    end
end
x0 = (θ = [2.7,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
           0.078,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
           0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
           1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
           ],
           Ω = PDMat(diagm(0 => [5.55,0.515])),
           σ_add = 0.388
           #σ_prop = 0.3
           )

laplace_estimated_params = (θ = [1.68975E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                          8.54637E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                          3.95757E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                          2.11952E+00 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
                          ],
                          Ω = PDMat(diagm(0 => [1.596,2.27638e-01])),
                          σ_add = 5.14457E-01)
# Elapsed estimation time in seconds:     0.23
# Elapsed covariance time in seconds:     0.17

@test_broken begin
  theopmodel_laplace_f(η,subject) = PuMaS.penalized_conditional_nll(theopmodel_laplace,subject, laplace_estimated_params, (η=η,))
  ηstar = [Optim.optimize(η -> theopmodel_laplace_f(η,theopp[i]),zeros(2),BFGS()).minimizer for i in 1:length(theopp)]
  #@test ηstar ≈  atol = 1e-3

  η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(theopmodel_laplace,subject,x0,(η=zeros(2),),Laplace()), theopp.subjects)
  #@test η0_mll ≈

  ml = sum(i -> PuMaS.marginal_nll_nonmem(theopmodel_laplace,theopp[i],x0,(η=ηstar[i],),Laplace()), 1:length(theopp))
  #@test ml ≈ rtol = 1e-6

  laplace_obj = 123.76439574418291

  function full_ll(θ)
    _x0 = (θ=θ,Ω = PDMat(diagm(0 => [5.55,0.515])),
               σ_add = 0.388)
    ηstar = [Optim.optimize(η -> theopmodel_laplace_f(η,theopp[i]),zeros(2),BFGS()).minimizer for i in 1:length(theopp)]
    sum(i -> PuMaS.marginal_nll_nonmem(theopmodel_laplace,theopp[i],
        _x0,(η=ηstar[i],),Laplace()), 1:10)
  end

  Optim.optimize(full_ll,[2.7,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
             0.078,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
             0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
             1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
             ],BFGS())
end

# run6.mod Laplace with interaction, diagonal omega and additive + proportional error , $COV = sandwich matrix
# method = laplacei
theopmodel_laplacei = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=[0.1,0.0008,0.004,0.1], init=[2.77,0.0781,0.0363,1.5], upper =[5,0.5,0.9,5])
        Ω ∈ PSDDomain(2)
        σ_add ∈ RealDomain(init=0.388)
        σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka = SEX == 1 ? θ[1] + η[1] : θ[4] + η[1]
        K = θ[2]
        CL  = θ[3]*WT + η[2]
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
x0 = (θ = [2.7,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
           0.078,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
           0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
           1.5 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
           ],
           Ω = PDMat(diagm(0 => [5.55,0.515])),
           σ_add = 0.388,
           σ_prop = 0.3
           )

laplacei_obj = 116.97275684239327
laplacei_estimated_params = (θ = [1.60941E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                           8.55663E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                           3.97472E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                           2.05830E+00 #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
                           ],
                           Ω_11  = 1.48117E+00,
                           Ω_21  = 0.00000E+00,
                           Ω_22  = 2.67215E-01,
                           σ_add = 1.88050E-01,
                           σ_prop = 1.25319E-02)
                          # Elapsed estimation time in seconds:     0.30
                         #  Elapsed covariance time in seconds:     0.32
