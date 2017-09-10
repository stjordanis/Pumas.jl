using PKPDSimulator
using DataFrames

# Load data
z = readtable(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/oral1_1cpt_KAVCL_MD_data.txt"), separator=' ')

# Define the ODE

function depot_model(t,u,p,du)
 Depot,Central = u
 Ka,CL,V = p
 du[1] = -Ka*Depot
 du[2] =  Ka*Depot - (CL/V)*Central
end
f = ParameterizedFunction(depot_model,[2.0,20.0,100.0])

# User definition of the set_parameters! function

function set_parameters!(p,u0,θ,η,z,i)
  col = getθ!(p, z, i)
  p[2] *= exp(η[1])
  p[3] *= exp(η[2])
  u0[1] = float(first(z[col, :][:amt]))
end

function getθ!(dest,z,i)
    col = findfirst(x->x==i, z[:id])
    d_n = z[col,:]
    for (i,para) ∈ enumerate((:ka, :cl, :v))
        dest[i] = float(first(d_n[para]))
    end
    col
end

#############################################
# The DiscreteCallback

function get_tstops(z)
    d_1 = nth_patient(z,1)
    cols = find(x->x==1, d_1[:evid])
    target_time = d_1[:time][cols]
end

## Finish the ODE Definition

tspan = (0.0,300.0)
u0 = zeros(2)
prob = ODEProblem(f,u0,tspan,callback=nth_patient_cb(z,1))
ts = get_tstops(z)

############################################
# Population setup

θ = [2.268,74.17,468.6,0.5876] # Not used in this case
ω = [0.05 0.0
     0.0 0.2]

#############################################
# Call simulate

sol = simulate(prob,set_parameters!,θ,ω,z;tstops=ts)

#=
using Plots; pyplot()
plot(sol,title="Plot of all trajectories",xlabel="time")
summ = MonteCarloSummary(sol,0:0.2:300)
plot(summ,title="Summary plot",xlabel="time")
=#
