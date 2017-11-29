using PKPDSimulator, NamedTuples, Distributions

# Read the data
covariates = [:sex,:wt,:etn]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,dvs,separator=',')

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
  @NT(Ka = θ[1],
      CL = θ[2]*((z.wt/70)^0.75)*(θ[4]^z.sex)*exp(η[1]),
      V  = θ[3]*exp(η[2]))
end

# Call simulate
prob = ODEProblem(depot_model,zeros(2),(0.0,19.0))
pkpd = PKPDModel(prob,set_parameters)

# Simulate individual 1
η1 = zeros(2)
sol1 = simulate(pkpd,θ,η1,data[1])

# Simulate Population
# testing turning off parallelism
sol = simulate(prob,set_parameters,θ,ω,data;parallel_type=:none)



#=
using Plots; plotly()
plot(sol,title="Plot of all trajectories",xlabel="time")
summ = MonteCarloSummary(sol,0:0.1:19)
plot(summ,title="Summary plot",xlabel="time")
=#

function reduction(sol,p,datai)
  sol(datai.obs_times;idxs=2)./p.V,false
end

pkpd = PKPDModel(prob,set_parameters,reduction)
sol1 = simulate(pkpd,θ,η1,data[1])

# Simulate individual 1 with reduction
sol1 = simulate(prob,set_parameters,θ,η1,data[1],reduction)

# Simulate population with reduction
sol = simulate(prob,set_parameters,θ,ω,data,reduction)

adderr(uij, θ) = Normal(uij, θ[end])

full = FullModel(pkpd, Independent(adderr))

σ = (0.025,)

# Simulate individual 1 with reduction and error model
sol1 = simulate(full,θ,η1,data[1])

# Simulate population with reduction and error model
#sol = simulate(prob,set_parameters,θ,ω,data,reduction,ϵ,error_model)

sol1 = simulate(full,θ,ω,data)
