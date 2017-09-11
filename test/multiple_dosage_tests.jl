using PKPDSimulator

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
z = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/oral1_1cpt_KAVCL_MD_data.txt"), covariates,dvs,
              separator=' ')

# Define the ODE

function depot_model(t,u,p,du)
 Depot,Central = u
 Ka,CL,V = p
 du[1] = -Ka*Depot
 du[2] =  Ka*Depot - (CL/V)*Central
end
f = ParameterizedFunction(depot_model,[2.0,20.0,100.0])

# User definition of the set_parameters! function

function set_parameters!(p,u0,θ,η,zi)
  p[1] = zi.covariates[:ka]
  p[2] = zi.covariates[:cl]*exp(η[1])
  p[3] = zi.covariates[:v]*exp(η[2])
end

# Population setup

θ = zeros(1) # Not used in this case
ω = [0.05 0.0
     0.0 0.2]

# Call simulate
tspan = (0.0,300.0)
num_dependent = 2
sol = simulate(f,tspan,num_dependent,set_parameters!,θ,ω,z)

#=
using Plots; pyplot()
plot(sol,title="Plot of all trajectories",xlabel="time")
summ = MonteCarloSummary(sol,0:0.2:300)
plot(summ,title="Summary plot",xlabel="time")
=#
