using PKPDSimulator, NamedTuples, Distributions

# Read the data
covariates = [:sex,:wt,:etn]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,dvs,separator=',')

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
prob = OneCompartmentModel(19.0)

# Simulate individual 1
η1 = zeros(2)
sol1 = simulate(prob,set_parameters,θ,η1,data[1])

# Simulate Population
sol = simulate(prob,set_parameters,θ,ω,data)

function reduction(sol,p,datai)
  sol(datai.obs_times)./p.V,false
end

# Simulate individual 1 with reduction
sol1 = simulate(prob,set_parameters,θ,η1,data[1],reduction)

# Simulate population with reduction
sol = simulate(prob,set_parameters,θ,ω,data,reduction)

function error_model(sol,η,ϵ)
  sol.*exp.(ϵ)
end
ϵ = Normal(0,0.025)

# Simulate individual 1 with reduction and error model
sol1 = simulate(prob,set_parameters,θ,η1,data[1],reduction,ϵ,error_model)

# Simulate population with reduction and error model
sol = simulate(prob,set_parameters,θ,ω,data,reduction,ϵ,error_model)
