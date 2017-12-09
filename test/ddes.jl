using PKPDSimulator, Base.Test, NamedTuples, DelayDiffEq

# Gut dosing model
function f(t,u,h,p,du)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central - 0.1h(t-1,Val{0},2)
end

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]))
end

h(t,idxs=0) = 0.0
prob = DDEProblem(f,h,zeros(2),(0.0,72.0),[1])
pkpd = PKPDModel(prob,set_parameters)
η = zeros(2)

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[0,0], time=[0,12],  cmt=[1,1])
sol  = simulate(pkpd,θ,η,data,MethodOfSteps(Tsit5()))

#data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[1,2], time=[0,12],  cmt=[2,2])
#sol  = simulate(pkpd,θ,η,data,MethodOfSteps(Tsit5()))
