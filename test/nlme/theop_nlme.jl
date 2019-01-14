using Test
using PuMaS, LinearAlgebra, Optim, StaticArrays

theopp_nlme = process_nmtran(example_nmtran_data("THEOPP"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 2

mdsl2 = @model begin
    @param begin
        θ ∈ VectorDomain(3,init=[3.24467e+01, 8.72879e-02, 1.49072e+00])
        Ω ∈ PSDDomain(@SMatrix([ 1.93973e-02  1.20854e-02  5.69131e-02
                                 1.20854e-02  2.02375e-02 -6.47803e-03
                                 5.69131e-02 -6.47803e-03  4.34671e-01]))
        Σ ∈ PSDDomain(Diagonal(@SVector([1.70385e-02, 8.28498e-02])))
    end

    @random begin
        η ~ Gaussian(Ω)
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
@test PuMaS.marginal_nll_nonmem(mdsl2,theopp_nlme,x0,Laplace()) ≈ 93.64166638742198 rtol = 1e-6 # NONMEM result
