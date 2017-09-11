using PKPDSimulator

# Read the data
covariates = [:sex,:wt,:etn]
dvs = [:dv]
z = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,dvs)

# Define the ODE

# Option 1: The macro
using ParameterizedFunctions
f = @ode_def_nohes DepotModel begin
  dDepot = -Ka*Depot
  dCentral = Ka*Depot - (CL/V)*Central
end Ka=>2.0 CL=>20.0 V=>100.0

# Option 2: The function
function depot_model(t,u,p,du)
 Depot,Central = u
 Ka,CL,V = p
 du[1] = -Ka*Depot
 du[2] =  Ka*Depot - (CL/V)*Central
end
f = ParameterizedFunction(depot_model,[2.0,20.0,100.0])

# Population parameters

θ = [2.268,74.17,468.6,0.5876] # [2.0,3.0,10.0,1.0] # Unfitted
ω = [0.05 0.0
     0.0 0.2]

# User definition of the set_parameters! function

function set_parameters!(p,u0,θ,η,zi)
  wt,sex = zi.covariates[:wt],zi.covariates[:sex]
  Ka = θ[1]
  CL = θ[2]*((wt/70)^0.75)*(θ[4]^sex)*exp(η[1])
  V  = θ[3]*exp(η[2])
  p[1] = Ka; p[2] = CL; p[3] = V
end

# Call simulate
tspan = (0.0,19.0)
num_dependent = 2
sol = simulate(f,tspan,num_dependent,set_parameters!,θ,ω,z)

#=
using Plots; pyplot()
plot(sol,title="Plot of all trajectories",xlabel="time")
summ = MonteCarloSummary(sol,0:0.1:19)
plot(summ,title="Summary plot",xlabel="time")
=#

output_func = function (sol,i)
  sol(z[i].obs_times;idxs=2)./sol.prob.f.params[3],false
end
sol = simulate(f,tspan,num_dependent,set_parameters!,θ,ω,z,output_func;saveat=z[1].obs_times)

# Notice the issue at events is that sol(0.0) is not well-defined and it takes
# the left-endpoint. Need a toggle for the right endpoint
