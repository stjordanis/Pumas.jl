using PKPDSimulator, NamedTuples, Distributions, DiffEqSensitivity,
      StaticArrays

# Read the data
covariates = [:sex,:wt,:etn]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,separator=',')

# Define the ODE

function depot_model(du,u,p,t)
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
  @SVector [θ[1],
            θ[2]*((z.wt/70)^0.75)*(θ[4]^z.sex)*exp(η[1]),
            θ[3]*exp(η[2])]
end

# Call simulate
prob = ODELocalSensitivityProblem(depot_model,zeros(2),(0.0,19.0),set_parameters(θ,η1,data[1].z))
pkpd = PKPDModel(prob,set_parameters)

# Simulate individual 1
η1 = zeros(2)
sol1 = simulate(pkpd,θ,η1,data[1])
