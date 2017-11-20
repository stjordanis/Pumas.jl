using PKPDSimulator, NamedTuples, Distributions

# Read the data
covariates = [:sex,:wt,:etn]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,dvs)

# Define the Analytical Solution

function depot_model(t,t0,u0,D,p)
  exp(-t*(p.Ka+p.CL/p.V))*(D*exp(p.Ka*t0+p.CL*t/p.V)*p.Ka*p.V +
  exp(t*p.Ka+p.CL*t0/p.V)*(u0*p.CL - (u0 + D)*p.Ka*p.V))/(p.CL-p.Ka*p.V)
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
prob = AnalyticalProblem(depot_model,0.0,(0.0,19.0))

# Simulate individual 1
η1 = zeros(2)
sol1 = simulate(prob,set_parameters,θ,η1,data[1])

using Plots; plotly()
plot(sol1,ylims=(0.0,3e4))
