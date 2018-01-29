using PKPDSimulator, Base.Test, NamedTuples

# Gut dosing model
function f(du,u,p,t)
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

###############################
# Test 15
###############################

data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[1,2], time=[0,12],  cmt=[2,2])

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

sol = get_sol(θ,data,abstol=1e-14,reltol=1e-14)

obs_times = [i*12 for i in 0:1]
res = 1000sol(obs_times+1e-14;idxs=2)/θ[3]
@test norm(res - [605.3220736386598;1616.4036675452326]) < 1e-8

###############################
# Test 16
###############################

data = build_dataset(amt=[10,20,10], ii=[24,24,24], addl=[0,0,0], ss=[1,2,1], time=[0,12,24],  cmt=[2,2,2])

sol = get_sol(θ,data,abstol=1e-14,reltol=1e-14)
obs_times = [i*12 for i in 0:5]
res = 1000sol(obs_times+1e-14;idxs=2)/θ[3]

true_res = [605.3220736386598
            1616.4036675452326
            605.3220736387212
            405.75952026789673
            271.98874030537564
            182.31950492267478]

@test norm(res - true_res) < 1e-9
