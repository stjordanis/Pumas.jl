using PKPDSimulator, NamedTuples, Base.Test

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/oral1_1cpt_KAVCL_MD_data.txt"), covariates,dvs)

# Define the ODE
function depot_model(du,u,p,t)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

# User definition of the set_parameters! function
function set_parameters(θ,η,z)
  @NT(Ka = z[:ka], CL = z[:cl], V = z[:v])
end
prob = ODEProblem(depot_model,zeros(2),(0.0,300.0))
pkpd = PKPDModel(prob,set_parameters)

# Population setup

θ = zeros(1) # Not used in this case
ω = zeros(2)

# Call simulate
sol1 = simulate(pkpd,θ,ω,data[2],abstol=1e-14,reltol=1e-14) # evid2
sol1 = simulate(pkpd,θ,ω,data[1],abstol=1e-14,reltol=1e-14)
sol = simulate(pkpd,θ,ω,data,abstol=1e-14,reltol=1e-14)
@test maximum(sol[1](0:0.1:300) - sol1(0:0.1:300)) < 1e-10

#=
using Plots; plotly()
plot(sol,title="Plot of all trajectories",xlabel="time")
summ = MonteCarloSummary(sol,0:0.2:300)
plot(summ,title="Summary plot",xlabel="time")
=#
