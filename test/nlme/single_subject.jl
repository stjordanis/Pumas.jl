using Test
using Pumas, LinearAlgebra

@testset "Single subject" begin
data = read_pumas(example_nmtran_data("sim_data_model1"))
#-----------------------------------------------------------------------# Test 1
mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(1, init=[0.5])
        Σ ∈ ConstDomain(0.1)
    end

    @pre begin
        CL = θ[1]
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

param = init_param(mdsl1)

for approx in (Pumas.FO, Pumas.FOI, Pumas.FOCE, Pumas.FOCEI, Pumas.Laplace, Pumas.LaplaceI, Pumas.HCubeQuad)
    @test_throws ArgumentError fit(mdsl1, data[1], param, approx())
end

end
@testset "Readme roundtrip" begin
model = @model begin
    @param   begin
    tvcl ∈ RealDomain(lower=0)
    tvv ∈ RealDomain(lower=0)
    pmoncl ∈ RealDomain(lower = -0.99)
    σ_prop ∈ RealDomain(lower=0)
    end

    @covariates wt isPM

    @pre begin
    CL = tvcl * (1 + pmoncl*isPM) * (wt/70)^0.75
    V  = tvv * (wt/70)
    end

    @dynamics ImmediateAbsorptionModel
    #@dynamics begin
    #    Central' =  - (CL/V)*Central
    #end

    @derived begin
      cp = @. 1000*(Central / V)
      dv ~ @. Normal(cp, sqrt(cp^2*σ_prop))
    end
end
ev = DosageRegimen(100, time=0, addl=4, ii=24)
param = (
  tvcl = 4.0,
  tvv  = 70,
  pmoncl = -0.7,
  σ_prop = 0.04
  )
choose_covariates() = (isPM = rand([1, 0]),  wt = rand(55:80))
pop_with_covariates = Population(map(i -> Subject(id=i, evs=ev, cvs=choose_covariates()),1:1000))
obs = simobs(model, pop_with_covariates, param, obstimes=0:1:120)

simdf = DataFrame(obs)
simdf.cmt .= 1

data = read_pumas(simdf, time=:time,cvs=[:isPM, :wt])

res = fit(model,data[1],param)
trf = Pumas.totransform(model.param)
fits = [fit(model, dat, param) for dat in data]
fitone = fit(model, first(data), param)
fitnp = fit(model, data, param, Pumas.NaivePooled())
fit2s = fit(model, data, param, Pumas.TwoStage())

# test something?

end
