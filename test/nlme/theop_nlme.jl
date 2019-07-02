using Test
using PuMaS, LinearAlgebra

theopp_nlme = read_pumas(example_nmtran_data("THEOPP"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 2

mdsl2 = @model begin
    @param begin
        θ ∈ VectorDomain(3,init=[3.24467E+01, 8.72879E-02, 1.49072E+00])
        Ω ∈ PSDDomain(init=Matrix{Float64}([ 1.93973E-02  1.20854E-02  5.69131E-02
                                             1.20854E-02  2.02375E-02 -6.47803E-03
                                             5.69131E-02 -6.47803E-03  4.34671E-01]))
        Σ ∈ PDiagDomain(init=[1.70385E-02, 8.28498E-02])
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        V  = θ[1] * exp(η[1])
        Ke = θ[2] * exp(η[2])
        Ka = θ[3] * exp(η[3])
        CL = Ke * V
    end

    @vars begin
        conc = Central / V
    end

    @dynamics OneCompartmentModel

    @derived begin
        dv ~ @. Normal(conc,sqrt(conc^2 *Σ.diag[1] + Σ.diag[end])+eps())
    end
end

param = init_param(mdsl2)
@test @inferred(deviance(mdsl2, theopp_nlme, param, PuMaS.LaplaceI())) ≈ 93.64166638742198 rtol = 1e-6 # NONMEM result
@test fit(mdsl2, theopp_nlme, param, PuMaS.FOCE()) isa PuMaS.FittedPuMaSModel
@test ηshrinkage(mdsl2, theopp_nlme, param, PuMaS.FOCEI()) ≈ [0.0161871, 0.0502453, 0.0133019] rtol = 1e-5
@test ϵshrinkage(mdsl2, theopp_nlme, param, PuMaS.FOCEI()) ≈ 0.09091845 rtol = 1e-6
ϵshrinkage(mdsl2,theopp_nlme, param, PuMaS.FOCE(),[empirical_bayes(mdsl2,subject,param,PuMaS.FOCEI()) for subject in theopp_nlme])
param = fit(mdsl2, theopp_nlme, param, PuMaS.FOCE()).param
@test ϵshrinkage(mdsl2, theopp_nlme, param, PuMaS.FOCEI(),[empirical_bayes(mdsl2,subject,param,PuMaS.FOCE()) for subject in theopp_nlme]) ≈ 0.4400298 rtol = 1e-3
@test ϵshrinkage(mdsl2, theopp_nlme, param, PuMaS.FOCE()) ≈ 0.1268684 rtol = 1e-3
@test aic(mdsl2, theopp_nlme, param, PuMaS.FOCEI()) ≈ 477.5715543243326 rtol = 1e-3 #regression test
@test bic(mdsl2, theopp_nlme, param, PuMaS.FOCEI()) ≈ 509.2823754727827 rtol = 1e-3 #regression test
param = init_param(mdsl2)
randeffs = [empirical_bayes(mdsl2,subject,param,PuMaS.FOCEI()) for subject in theopp_nlme]
@test [PuMaS.ipred(mdsl2, subject, param, randeff) for (subject,randeff) in zip(theopp_nlme,randeffs)]   isa Vector # FIXME! come up with a better test
@test [PuMaS.cipred(mdsl2, subject, param, randeff) for (subject,randeff) in zip(theopp_nlme,randeffs)]  isa Vector # FIXME! come up with a better test
@test [PuMaS.cipredi(mdsl2, subject, param, randeff) for (subject,randeff) in zip(theopp_nlme,randeffs)] isa Vector # FIXME! come up with a better test
