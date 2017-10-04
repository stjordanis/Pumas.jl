using PKPDSimulator, NamedTuples

# Read the data
covariates = [:sex,:wt,:etn]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,dvs)

# Define the ODE

function depot_model(t,u,p,du)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

# Population parameters

θ = [2.268,74.17,468.6,0.5876] # [2.0,3.0,10.0,1.0] # Unfitted
ω = [0.05 0.0
     0.0 0.2]

# User definition of the set_parameters! function

function set_parameters(θ,η,z)
  wt,sex = z[:wt],z[:sex]
  @NT(Ka = θ[1],
      CL = θ[2]*((wt/70)^0.75)*(θ[4]^sex)*exp(η[1]),
      V  = θ[3]*exp(η[2]))
end

# Call simulate
tspan = (0.0,19.0)
num_dependent = 2
sol = simulate(depot_model,tspan,num_dependent,set_parameters,θ,ω,data)

#=
using Plots; plotly()
plot(sol,title="Plot of all trajectories",xlabel="time")
summ = MonteCarloSummary(sol,0:0.1:19)
plot(summ,title="Summary plot",xlabel="time")
=#

function reduction(sol,p,datai)
  sol(datai.obs_times;idxs=2)./p.V,false
end

sol = simulate(depot_model,tspan,num_dependent,set_parameters,θ,ω,data,reduction)

function error_model(sol,ϵ)
  sol.*exp(ϵ)
end

σ = 0.025
sol = simulate(depot_model,tspan,num_dependent,set_parameters,θ,ω,data,reduction,σ,error_model)
