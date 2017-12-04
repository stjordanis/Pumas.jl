using PKPDSimulator, Base.Test, NamedTuples

# Gut dosing model
function f(t,u,p,du)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]))
end

function get_sol(θ,data;num_dv=2,kwargs...)
    prob = ODEProblem(f,zeros(num_dv),(0.0,72.0))
    pkpd = PKPDModel(prob,set_parameters)
    η = zeros(2)
    sol  = simulate(pkpd,θ,η,data;kwargs...)
end

data = build_dataset(amt=[10,20], ii=[24,24], addl=[1,1], ss=[1,2], time=[0,12],  cmt=[2,2])

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

sol = get_sol(θ,data)

obs_times = [i*12 for i in 0:5]
res = 1000sol(obs_times+1e-14;idxs=2)/θ[3]
