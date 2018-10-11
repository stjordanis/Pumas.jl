using PuMaS, Test, Distributions, DiffEqJump

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

params = ParamSet((θ=VectorDomain(3, lower=zeros(4),init=θ), Ω=PSDDomain(2)))
function randomfx(p)
  RandomEffectSet((η=MvNormal(p.Ω),))
end

function collate(params, randoms, covars)
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

prob = ODEProblem(f,zeros(2),(0.0,100.0))

rate(u,p,t) = .15u[1]
function affect!(integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
jump = VariableRateJump(rate,affect!)
jump_prob = JumpProblem(prob,Direct(),jump)

function err(params, randoms, covars, u,p, t)
    V = p.V
    Depot, Central = u
    Σ = params.Σ
    conc = Central / V
    (dv=Normal(conc, conc*Σ),)
end

init = (_param, _random, _covariates,_collate,t) -> [0.0,0.0]
post = (_param, _random, _covariates,_collate,_odevars,t) -> (conc = _odevars[2] / _collate.V,)
model = PuMaS.PKPDModel(params,randomfx,collate,init,prob,post,err)

x0 = init_param(model)
y0 = init_random(model, x0)

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12]))
sol  = solve(model,data,x0,y0,Tsit5())

#data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))
#sol  = simulate(pkpd,θ,η,data,Tsit5())
