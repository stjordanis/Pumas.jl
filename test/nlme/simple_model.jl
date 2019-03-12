using Test
using PuMaS, LinearAlgebra, Optim

data = process_nmtran(example_nmtran_data("sim_data_model1"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 1
mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(1,init=[0.5])
        Ω ∈ PDiagDomain(PDiagMat(fill(0.04, 1)))
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
        dv ~ @. Normal(conc,conc*sqrt(Σ)+eps())
    end
end

fixeffs = init_fixeffs(mdsl1)

PuMaS.randeffs_estimate(mdsl1, data[1], fixeffs, PuMaS.Laplace())

for (ηstar, dt) in zip([-0.1007, 0.0167, -0.0363, -0.0820, 0.1061, 0.0473, -0.1007, -0.0361, -0.0578, -0.0181], data)
    @test PuMaS.randeffs_estimate(mdsl1, dt, fixeffs, PuMaS.Laplace())[1] ≈ ηstar rtol=1e-2
end
for (ηstar, dt) in zip([-0.114654,0.0350263,-0.024196,-0.0870518,0.0750881,0.059033,-0.114679,-0.023992,-0.0528146,-0.00185361], data)
    @test PuMaS.randeffs_estimate(mdsl1, dt, fixeffs, PuMaS.LaplaceI())[1] ≈ ηstar rtol=1e-3
end

@test PuMaS.marginal_nll_nonmem(mdsl1, data, fixeffs, PuMaS.FOCEI())    ≈ 56.410938825140313 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, fixeffs, PuMaS.FOCE())     ≈ 56.476216665029462 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, fixeffs, PuMaS.FO())       ≈ 56.474912258255571 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, fixeffs, PuMaS.Laplace())  ≈ 56.613069180382027 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, fixeffs, PuMaS.LaplaceI()) ≈ 56.810343602063618 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, fixeffs, [0.0],PuMaS.LaplaceI()) ≈ 57.19397077905644 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, fixeffs, (η=[0.0],),PuMaS.LaplaceI()) ≈ 57.19397077905644 rtol=1e-6

@test fit(mdsl1, data, init_fixeffs(mdsl1), PuMaS.LaplaceI(),
    optimmethod=BFGS(linesearch=Optim.LineSearches.BackTracking()), optimautodiff=:finite) isa PuMaS.FittedPuMaSModel
@test fit(mdsl1, data, init_fixeffs(mdsl1), PuMaS.LaplaceI(),
    optimmethod=Newton(linesearch=Optim.LineSearches.BackTracking()), optimautodiff=:finite) isa PuMaS.FittedPuMaSModel
@test fit(mdsl1, data, init_fixeffs(mdsl1), PuMaS.LaplaceI(),
    optimmethod=BFGS(linesearch=Optim.LineSearches.BackTracking()), optimautodiff=:forward) isa PuMaS.FittedPuMaSModel
@test fit(mdsl1, data, init_fixeffs(mdsl1), PuMaS.LaplaceI(),
    optimmethod=Newton(linesearch=Optim.LineSearches.BackTracking()), optimautodiff=:forward) isa PuMaS.FittedPuMaSModel
