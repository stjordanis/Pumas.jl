using PuMaS, Test, DiffEqJump

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

p = ParamSet((θ=VectorDomain(3, lower=zeros(4),init=θ), Ω=PSDDomain(2)))
function randomfx(p)
  ParamSet((η=Gaussian(p.Ω),))
end

function pre_f(params, randoms, covars)
    θ = params.θ
    η = randoms.η
    (Ka = θ[1],
     CL = θ[2]*exp(η[1]),
     V  = θ[3]*exp(η[2]))
end

function ode_f(du,u,p,t)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

prob = ODEProblem(ode_f,nothing,nothing)

rate(u,p,t) = .15u[1]
function affect!(integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
jump = VariableRateJump(rate,affect!)
jump_prob = JumpProblem(prob,Direct(),jump)

init_f = (col,t) -> [0.0,0.0]

function derived_f(col,sol,obstimes)
    central = map(x->x[2], sol)
    conc = @. central / col.V

    ___dv = @. Normal(conc, conc*col.Σ)
    dv = @. rand(___dv)
    (obs_cmax = maximum(dv),
     T_max = maximum(obstimes),
     dv=dv), (dv=___dv,)
end

model = PuMaS.PKPDModel(p,randomfx,pre_f,init_f,jump_prob,derived_f)

x0 = init_param(model)
y0 = init_random(model, x0)

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12]))
sol  = solve(model,data,x0,y0,Tsit5())

#data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))
#sol  = simulate(pkpd,θ,η,data,Tsit5())

using Plots
plot(sol)
