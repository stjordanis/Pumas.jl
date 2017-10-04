using PKPDSimulator, NamedTuples

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/oral1_1cpt_KAVCL_MD_data.txt"), covariates,dvs,
              separator=' ')

# Define the ODE
function depot_model(t,u,p,du)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

# User definition of the set_parameters! function
function set_parameters(θ,η,z)
  @NT(Ka = z[:ka], CL = z[:cl], V = z[:v])
end

# Population setup

θ = zeros(1) # Not used in this case
ω = zeros(2)

# Call simulate
tspan = (0.0,300.0)
num_dependent = 2
sol = simulate(depot_model,tspan,num_dependent,set_parameters,θ,ω,data)

#=
using Plots; plotly()
plot(sol,title="Plot of all trajectories",xlabel="time")
summ = MonteCarloSummary(sol,0:0.2:300)
plot(summ,title="Summary plot",xlabel="time")
=#
