using Test
using PuMaS, LinearAlgebra, Optim

data = process_nmtran(example_nmtran_data("sim_data_model1"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 1
tdist = @model begin
    @param begin
        θ ∈ VectorDomain(1,init=[0.5])
        Ω ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        Σ ∈ ConstDomain(0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ[1] * exp(η[1])
        V  = 1.0
    end

    @vars begin
        conc = Central / V
    end

    @dynamics ImmediateAbsorptionModel

    @derived begin
        dv ~ @. LocationScale(conc, conc*sqrt(Σ)+eps(), TDist(30))
    end
end


fixeffs = init_param(tdist)

for (d, v) in zip(data, [-0.110095703125000, 0.035454025268555, -0.024982604980469,
                         -0.085317642211914, 0.071675025939941, 0.059612709045410,
                         -0.110117317199707, -0.024778854370117, -0.053085464477539,
                         -0.002428230285645])
  @test PuMaS.rfx_estimate(tdist, d, fixeffs, PuMaS.LaplaceI())[1] ≈ v rtol=1e-6
end

@test_broken PuMaS.marginal_nll_nonmem(tdist, data, fixeffs, PuMaS.FOCEI()) ≈ -10000 rtol = 1e-6 # Need to get the right value of the mll
@test_broken PuMaS.marginal_nll_nonmem(tdist, data, fixeffs, PuMaS.FOCE())
@test PuMaS.marginal_nll_nonmem(tdist, data, fixeffs, PuMaS.LaplaceI()) ≈ 57.112537604068990 rtol=1e-6

function full_ll(θ)
  _fixeffs = (θ=θ, Ω=fill(0.04, 1, 1), Σ=0.1)
  PuMaS.marginal_nll_nonmem(tdist, data,_fixeffs, PuMaS.LaplaceI())
end

Optim.optimize(full_ll, [0.5], BFGS())
