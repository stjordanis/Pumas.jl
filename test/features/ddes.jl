using PuMaS, Test, DelayDiffEq

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

function f(du,u,h,p,t)
 Depot,Central = u
 Central_Delay = h(p,t-10)[2]
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central_Delay
end

init_f = (col,t) -> [0.0,0.0]
h(p,t) = zeros(2)

prob = DDEProblem(f,nothing,h,nothing,nothing)

function derived_f(col,sol,obstimes,subject)
    central = sol(obstimes;idxs=2)
    conc = @. central / col.V
    dv = @. Normal(conc, conc*col.Σ)
    dv = @. rand(___dv)
    (dv=dv,)
end

function observed_f(col,sol,obstimes,samples,subject)
    samples
end

model = PuMaS.PuMaSModel(p,randomfx,pre_f,init_f,prob,derived_f,observed_f)

param = init_param(model)
randeffs = init_randeffs(model, param)

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12]))
sol  = solve(model,data,param,randeffs,alg=MethodOfSteps(Tsit5()))

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))
sol  = solve(model,data,param,randeffs,alg=MethodOfSteps(Tsit5()))

# Regression test on interpolation issue
@test all(sol(24:0.5:30)[2,:] .< 53)
