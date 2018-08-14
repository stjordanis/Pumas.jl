using PuMaS, NamedTuples, Distributions, DiffEqSensitivity,
      StaticArrays

θ = [
   1.5,  #Ka
   1.0,  #CL
   30.0 #V
   ]

params = ParamSet(@NT(θ=VectorDomain(3, lower=zeros(4),init=θ), Ω=PSDDomain(2)))
function randomfx(p)
  RandomEffectSet(@NT(η=MvNormal(p.Ω)))
end

function collate(params, randoms, covars)
  θ = params.θ
  η = randoms.η
  @NT(Ka = θ[1],
      CL = θ[2]*exp(η[1]),
      V  = θ[3]*exp(η[2]))
end

function f(du,u,p,t)
  Depot,Central = u
  du[1] = -p.Ka*Depot
  du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

prob = ODEProblem(f,zeros(2),(0.0,100.0))

function err(params, randoms, covars, u,p, t)
  V = p.V
  Depot, Central = u
  Σ = params.Σ
  conc = Central / V
  @NT(dv=Normal(conc, conc*Σ))
end

init = (_param, _random, _data_cov,_collate,t) -> [0.0,0.0]
post = (_param, _random, _data_cov,_collate,_odevars,t) -> @NT(conc = _odevars[2] / _collate.V)
model = PuMaS.PKPDModel(params,randomfx,collate,init,prob,post,err)

x0 = init_param(model)
y0 = init_random(model, x0)
p = collate(x0, y0, data.covariates)

data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[0,0], time=[0,12],  cmt=[1,1])
prob2 = ODELocalSensitivityProblem(f,zeros(2),(0.0,19.0),p)
model2 = PuMaS.PKPDModel(params,randomfx,collate,init,prob2,post,err)
sol  = pkpd_solve(model2,data,x0,y0,Tsit5())

data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[1,2], time=[0,12],  cmt=[2,2])
prob2 = ODELocalSensitivityProblem(f,zeros(2),(0.0,19.0),p)
model2 = PuMaS.PKPDModel(params,randomfx,collate,init,prob2,post,err)
sol  = pkpd_solve(model2,data,x0,y0,Tsit5())
