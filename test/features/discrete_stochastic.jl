using PuMaS, Test, DiffEqJump

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

p = ParamSet((θ=VectorDomain(3, lower=zeros(4),init=θ), Ω=PSDDomain(2)))
function randomfx(p)
  ParamSet((η=MvNormal(p.Ω),))
end

function pre_f(params, randoms, subject)
    θ = params.θ
    η = randoms.η
    (Ka = θ[1],
     CL = θ[2]*exp(η[1]),
     V  = θ[3]*exp(η[2]))
end

prob = DiscreteProblem([0.0,0.0],(0.0,72.0))

rate1(u,p,t) = 100p.Ka*u[1]
function affect1!(integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
jump1 = ConstantRateJump(rate1,affect1!)

rate2(u,p,t) = (p.CL/p.V)*u[2]
function affect2!(integrator)
  integrator.u[2] -= 1
end
jump2 = ConstantRateJump(rate2,affect2!)
jump_prob = JumpProblem(prob,Direct(),jump1,jump2)

init_f = (col,t) -> [0.0,0.0]

function derived_f(col,sol,obstimes,subject)
    central = sol(obstimes;idxs=2)
    conc = @. central / col.V
    dv = @. Normal(conc, conc*col.Σ)
    (dv=dv,)
end

model = PuMaS.PuMaSModel(p,randomfx,pre_f,init_f,jump_prob,derived_f)

param = init_param(model)
randeffs = init_randeffs(model, param)

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12]))
sol  = solve(model,data,param,randeffs,FunctionMap())
