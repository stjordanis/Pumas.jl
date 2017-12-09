using PKPDSimulator, Base.Test, NamedTuples, DiffEqJump

# Gut dosing model
function f(t,u,p,du)
 Depot,Central = u
 du[2] =  - (p.CL/p.V)*Central
end

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]))
end

rate(t,u) = .15u[1]
function affect!(integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
jump = VariableRateJump(rate,affect!)
prob = ODEProblem(f,zeros(2),(0.0,100.0))
jump_prob = JumpProblem(prob,Direct(),jump)
pkpd = PKPDModel(jump_prob,set_parameters)
η = zeros(2)

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[0,0], time=[0,12],  cmt=[1,1])
sol  = simulate(pkpd,θ,η,data,Tsit5())

data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[1,2], time=[0,12],  cmt=[1,1])
sol  = simulate(pkpd,θ,η,data,Tsit5())
