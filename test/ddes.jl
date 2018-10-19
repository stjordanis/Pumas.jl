using PuMaS, Test, DelayDiffEq

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

params = ParamSet((θ=VectorDomain(3, lower=zeros(4),init=θ), Ω=PSDDomain(2)))
function randomfx(p)
  RandomEffectSet((η=MvNormal(p.Ω),))
end

function pre(params, randoms, covars)
    θ = params.θ
    η = randoms.η
    (Ka = θ[1],
     CL = θ[2]*exp(η[1]),
     V  = θ[3]*exp(η[2]))
end

function f(du,u,p,t)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

function g(t,u,du)
 du[1] = 0.5u[1]
 du[2] = 0.1u[2]
end

prob = SDEProblem(f,g,zeros(2),(0.0,72.0))

function err(params, randoms, covars, u,p, t)
    V = p.V
    Depot, Central = u
    Σ = params.Σ
    conc = Central / V
    (dv=Normal(conc, conc*Σ),)
end

init = (_param, _random, _covariates,_pre,t) -> [0.0,0.0]
post = (_param, _random, _covariates,_pre,_odevars,t) -> (conc = _odevars[2] / _pre.V,)
model = PuMaS.PKPDModel(params,randomfx,pre,init,prob,post,err)

x0 = init_param(model)
y0 = init_random(model, x0)

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12]))
sol  = solve(model,data,x0,y0,MethodOfSteps(Tsit5()))

#data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))
#sol  = simulate(pkpd,θ,η,data,MethodOfSteps(Tsit5()))
