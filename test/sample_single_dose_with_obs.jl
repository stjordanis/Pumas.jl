using PKPDSimulator, NamedTuples, Distributions

# Read the data
covariates = [:v,:cl,:ka]
dvs = [:dv]
path = joinpath(Pkg.dir("PKPDSimulator"),"examples/oral1_1cpt_KAVCL_SD_data.txt")
data = process_data(path,
       covariates,dvs,separator = ' ')

# Define the ODE

function depot_model(t,u,p,du)
  Depot,Central = u
  du[1] = -p.Ka*Depot
  du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

# Population parameters

θ = [2.268,74.17,468.6] # [2.0,3.0,10.0,1.0] # Unfitted
ω = [0.05 0.0
     0.0 0.2]

# User definition of the set_parameters! function

function set_parameters(θ,η,z)
  @NT(Ka = z[:ka],
      CL = z[:cl],#*exp(η[1]),
      V  = z[:v])#*exp(η[2]))
end

# Call simulate
prob = ODEProblem(depot_model,zeros(2),(0.0,72.0))

function reduction(sol,p,datai)
  sol(datai.obs_times;idxs=2)./p.V
end

sol = simulate(prob,set_parameters,θ,ω,data,reduction,abstol=1e-12,reltol=1e-12)
