using Test
using PuMaS, LinearAlgebra, Optim

theopp_nlme = process_nmtran(example_nmtran_data("THEOPP"))

#likelihood tests from NLME.jl 
#-----------------------------------------------------------------------# Test 2

mdsl2 = @model begin
    @param begin
        θ ∈ VectorDomain(3,init=[3.24467E+01, 8.72879E-02, 1.49072E+00])
        Ω ∈ PSDDomain(Matrix{Float64}([ 1.93973E-02  1.20854E-02  5.69131E-02
                                        1.20854E-02  2.02375E-02 -6.47803E-03
                                        5.69131E-02 -6.47803E-03  4.34671E-01]))
        Σ ∈ PSDDomain(Diagonal([1.70385E-02, 8.28498E-02]))
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
        dv ~ @. Normal(conc,sqrt(conc^2 *Σ[1] + Σ[end])+eps())
    end
end

x0 = init_param(mdsl2)
mdsl_f(η,subject) = PuMaS.penalized_conditional_nll(mdsl2,subject, x0, (η=η,))
ηstar = [Optim.optimize(η -> mdsl_f(η,theopp_nlme[i]),zeros(3),BFGS()).minimizer for i in 1:12]
ml = sum(i -> PuMaS.marginal_nll_nonmem(mdsl2,theopp_nlme[i],x0,(η=ηstar[i],),Laplace()), 1:12)
@test ml ≈ 93.64166638742198 rtol = 1e-6 # NONMEM result