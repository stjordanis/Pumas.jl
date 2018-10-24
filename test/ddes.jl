using PuMaS, Test, DelayDiffEq

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

p = ParamSet((θ=VectorDomain(3, lower=zeros(4),init=θ), Ω=PSDDomain(2)))
function randomfx(p)
  RandomEffectSet((η=MvNormal(p.Ω),))
end

function pre_f(params, randoms, covars)
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

function derived_f(col,sol,obstimes)
    central = map(x->x[2], sol)
    conc = @. central / col.V
    ___dv = @. Normal(conc, conc*col.Σ)
    dv = @. rand(___dv)
    (obs_cmax = maximum(dv),
     T_max = maximum(obstimes),
     dv=dv), (dv=___dv,)
end

model = PuMaS.PKPDModel(p,randomfx,pre_f,init_f,prob,derived_f)

x0 = init_param(model)
y0 = init_random(model, x0)

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12]))
sol  = solve(model,data,x0,y0,MethodOfSteps(Tsit5()))

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))
sol  = solve(model,data,x0,y0,MethodOfSteps(Tsit5()))
