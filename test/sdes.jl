using PKPDSimulator, Base.Test, NamedTuples, StochasticDiffEq

# Gut dosing model
function f(t,u,p,du)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

function g(t,u,du)
 du[1] = 0.5u[1]
 du[2] = 0.1u[2]
end

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]))
end

prob = SDEProblem(f,g,zeros(2),(0.0,72.0))
pkpd = PKPDModel(prob,set_parameters)
η = zeros(2)

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[0,0], time=[0,12],  cmt=[1,1])
sol  = simulate(pkpd,θ,η,data,SRIW1())

data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[1,2], time=[0,12],  cmt=[2,2])
sol  = simulate(pkpd,θ,η,data,SRIW1())
