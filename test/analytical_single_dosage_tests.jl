using PKPDSimulator, NamedTuples, Distributions

# Read the data
covariates = [:sex,:wt,:etn]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,dvs,separator=',')

# Population parameters

θ = [2.268,74.17,468.6,0.025] # [2.0,3.0,10.0,1.0] # Unfitted
ω = [0.05 0.0
     0.0 0.2]

# Build model

function set_parameters(θ,η,z)
  @NT(Ka = θ[1],
      CL = θ[2]*((z.wt/70)^0.75)*(θ[4]^z.sex)*exp(η[1]),
      V  = θ[3]*exp(η[2]))
end
pkpd = PKPDModel(OneCompartmentModel(19.0),set_parameters)

# Simulate individual 1
η1 = zeros(2)
sol1 = simulate(pkpd,θ,η1,data[1])

sol1(1.0)

# Simulate Population
sol = simulate(pkpd,θ,ω,data)

function reduction(sol,p,datai)
  sol(datai.obs.times;idxs=2)./p.V
end
pkpd = PKPDModel(OneCompartmentModel(19.0),set_parameters,reduction)

# Simulate individual 1 with reduction
sol1 = simulate(pkpd,θ,η1,data[1])

# Simulate population with reduction
sol = simulate(pkpd,θ,ω,data)

adderr(uij, θ) = Normal(uij, θ[end])
full = FullModel(pkpd, Independent(adderr))

# Simulate individual 1 with reduction and error model
sol1 = simulate(full,θ,η1,data[1])

# Simulate population with reduction and error model
sol = simulate(full,θ,ω,data)
