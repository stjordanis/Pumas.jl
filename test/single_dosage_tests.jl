using PKPDSimulator, NamedTuples, Distributions

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

# Simulate individual 1
η1 = zeros(2)
sol1 = simulate(depot_model,tspan,num_dependent,set_parameters,θ,η1,data[1])

# Simulate Population
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

# Simulate individual 1 with reduction
sol1 = simulate(depot_model,tspan,num_dependent,set_parameters,θ,η1,data[1],reduction)

# Simulate population with reduction
sol = simulate(depot_model,tspan,num_dependent,set_parameters,θ,ω,data,reduction)

function error_model(sol,η,ϵ)
  sol.*exp.(ϵ)
end

ϵ = Normal(0,0.025)

# Simulate individual 1 with reduction and error model
sol1 = simulate(depot_model,tspan,num_dependent,set_parameters,θ,η1,data[1],reduction,ϵ,error_model)

# Simulate population with reduction and error model
sol = simulate(depot_model,tspan,num_dependent,set_parameters,θ,ω,data,reduction,ϵ,error_model)
