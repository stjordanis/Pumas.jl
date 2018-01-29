using PKPDSimulator, NamedTuples, Distributions, StaticArrays, Base.Test

# Read the data
covariates = [:sex,:wt,:etn]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,separator=',')

# Define the ODE

function depot_model(u,p,t)
 Depot,Central = u
 @SVector [-p.Ka*Depot,
           p.Ka*Depot - (p.CL/p.V)*Central
           ]
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
prob = ODEProblem{false}(depot_model,zeros(SVector{2}),(0.0,19.0))
pkpd = PKPDModel(prob,set_parameters)

# Simulate individual 1
η1 = zeros(2)
sol1 = simulate(pkpd,θ,η1,data[1])

# Now allow steady states

data = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[1,2], time=[0,12],  cmt=[2,2])

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]))
end

sol = simulate(pkpd,θ,η1,data)

obs_times = [i*12 for i in 0:1]
res = 1000sol(obs_times+1e-12;idxs=2)/θ[3]
@test norm(res - [605.3220736386598;1616.4036675452326]) < 1e-3
