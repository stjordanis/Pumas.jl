using PuMaS, Test, StochasticDiffEq

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

p = ParamSet((θ=VectorDomain(3, lower=zeros(4),init=θ), Ω=PSDDomain(2)))
function randomfx(p)
  ParamSet((η=MvNormal(p.Ω),))
end

function pre_f(params, randoms, covars)
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

function g(du,u,p,t)
 du[1] = 0.5u[1]
 du[2] = 0.1u[2]
end

prob = SDEProblem(f,g,nothing,nothing)

init_f = (col,t) -> [0.0,0.0]

function derived_f(col,sol,obstimes)
    central = map(x->x[2], sol)
    conc = @. central / col.V
    dv = @. Normal(conc, conc*col.Σ)
    (dv=dv,)
end

function observed_f(col,sol,obstimes,samples)
    samples
end

model = PuMaS.PKPDModel(p,randomfx,pre_f,init_f,prob,derived_f,observed_f)

x0 = init_param(model)
y0 = init_random(model, x0)

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12]))
sol  = solve(model,data,x0,y0,alg=SRIW1(),tspan=(0.0,90.0))

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))
sol  = solve(model,data,x0,y0,alg=SRIW1())
