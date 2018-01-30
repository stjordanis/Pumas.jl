using PKPDSimulator, Base.Test, NamedTuples

function get_nonem_data(i)
    data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
                  "examples/event_data/ev$i.csv"), covariates,dvs,
                  separator=',')
    obsdata = process_data(joinpath(Pkg.dir("PKPDSimulator"),
                "examples/event_data","data$i.csv"),Symbol[],Symbol[:cp],
                separator=',')
    obs = obsdata.subjects[1].obs.vals[1]
    obs_times = obsdata.subjects[1].obs.times
    data,obs,obs_times
end

# Fake dvs and covariates for testing
covariates = [1]
dvs = [1]


# Indirect Response Model (irm1)

###############################
# Test 23
###############################

# ev23 - Testing irm1
# - Indirect response model, type 1
# - Inhibition of response input
# - Two-compartment PK model
# - Optional nonlinear clearance - not being used in this example (Vmax and Km)
# use ev23.csv in PKPDSimulator/examples/event_data/
# For the current example, a bolus dose is given into the gut compartment at time=0 and three additional doses
# every 24 hours.

# Two main response variables -
# - concentration in the central compartment
# - and the amount in the response compartment

# the predicted concentration in the central compartment inhibhits the
# production of the pharmacodynamic response variable

# NOTE: No scaling of concentrations is required
# A  100 mg dose is given into the gut compartment (cmt=1) at time zero with a bioav of 1 (bioav1)

# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=24: each additional dose is given with a frequency of ii=24 hours
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(23)

function f(du,u,p,t)
 ev1,cent,periph,resp = u
 cp     = (cent/p.Vc)
 ct     = (periph/p.Vp)
 CLNL   = p.Vmax/(p.Km+cp)
 INH    = (p.IMAX*cp^p.γ/(p.IC50^p.γ+cp^p.γ))

 du[1] = -p.Ka1*ev1
 du[2] =  p.Ka1*ev1 - (p.CL+CLNL+p.Q)*cp  + p.Q*ct
 du[3] =  p.Q*cp  - p.Q*ct
 du[4] =  p.Kin*(1-INH)  - p.Kout*resp
end

function set_parameters(θ,η,z)
    @NT(Ka1     = θ[1],
        CL      = θ[2]*exp(η[1]),
        Vc      = θ[3]*exp(η[2]),
        Q       = θ[4]*exp(η[3]),
        Vp      = θ[5]*exp(η[4]),
        Kin     = θ[6]*exp(η[5]),
        Kout    = θ[7]*exp(η[6]),
        IC50    = θ[8]*exp(η[7]),
        IMAX    = θ[9]*exp(η[8]),
        γ       = θ[10]*exp(η[9]),
        Vmax    = θ[11]*exp(η[10]),
        Km      = θ[12]*exp(η[11]))
end

θ = [
    1, # Ka1  Absorption rate constant 1 (1/time)
    1, # CL   Clearance (volume/time)
   20, # Vc   Central volume (volume)
    2, # Q    Inter-compartmental clearance (volume/time)
   10, # Vp   Peripheral volume of distribution (volume)
   10, # Kin  Response in rate constant (1/time)
    2, # Kout Response out rate constant (1/time)
    2, # IC50 Concentration for 50% of max inhibition (mass/volume)
    1, # IMAX Maximum inhibition
    1, # γ    Emax model sigmoidicity
    0, # Vmax Maximum reaction velocity (mass/time)
    2  # Km   Michaelis constant (mass/volume)
    ]

resp_0 = θ[6]/θ[7] # Kin/Kout
prob = ODEProblem(f,[0.0,0.0,0.0,resp_0],(0.0,240.0))
pkpd = PKPDModel(prob,set_parameters)
η = zeros(11)
sol  = simulate(pkpd,θ,η,data[1],abstol=1e-12,reltol=1e-12)

obsdata = process_data(joinpath(Pkg.dir("PKPDSimulator"),
            "examples/event_data","data23.csv"),Symbol[],Symbol[:ev1,:cp,:periph,:resp],
            separator=',')
obs_ev1s = obsdata.subjects[1].obs.vals[1]
obs_cps = obsdata.subjects[1].obs.vals[2]
obs_periphs = obsdata.subjects[1].obs.vals[3]
obs_resps = obsdata.subjects[1].obs.vals[4]

ev1s = sol(obs_times;idxs=1).u
@test maximum(ev1s - obs_ev1s) < 1e-6

cps = sol(obs_times;idxs=2)./θ[3]
@test maximum(cps - obs) < 1e-6

periphs = sol(obs_times;idxs=3).u
@test maximum(periphs - obs_periphs) < 1e-6

resps = sol(obs_times;idxs=4).u
@test maximum(resps - obs_resps) < 1e-6

# Indirect Response Model (irm2)

###############################
# Test 24
###############################

# ev24 - Testing irm2
# - Indirect response model, type 2
# - Inhibition of response elimination
# - Two-compartment PK model
# - Optional nonlinear clearance - not being used in this example (Vmax and Km)
# use ev24.csv in PKPDSimulator/examples/event_data/
# For the current example, a bolus dose is given into the gut compartment at time=0 and three additional doses
# every 24 hours.

# Two main response variables -
# - concentration in the central compartment
# - and the amount in the response compartment

# the predicted concentration in the central compartment inhibhits the elimination of the
# pharmacodynamic response variable

# NOTE: No scaling of concentrations is required
# A  100 mg dose is given into the gut compartment (cmt=1) at time zero with a bioav of 1 (bioav1)

# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=24: each additional dose is given with a frequency of ii=24 hours
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(24)

function f(du,u,p,t)
    ev1,cent,periph,resp = u
    cp     = (cent/p.Vc)
    ct     = (periph/p.Vp)
    CLNL   = p.Vmax/(p.Km+cp)
    INH    = (p.IMAX*cp^p.γ/(p.IC50^p.γ+cp^p.γ))

    du[1] = -p.Ka1*ev1
    du[2] =  p.Ka1*ev1 - (p.CL+CLNL+p.Q)*cp  + p.Q*ct
    du[3] =  p.Q*cp  - p.Q*ct
    du[4] =  p.Kin  - p.Kout*(1-INH)*resp
end

function set_parameters(θ,η,z)
   @NT(Ka1     = θ[1],
       CL      = θ[2]*exp(η[1]),
       Vc      = θ[3]*exp(η[2]),
       Q       = θ[4]*exp(η[3]),
       Vp      = θ[5]*exp(η[4]),
       Kin     = θ[6]*exp(η[5]),
       Kout    = θ[7]*exp(η[6]),
       IC50    = θ[8]*exp(η[7]),
       IMAX    = θ[9]*exp(η[8]),
       γ       = θ[10]*exp(η[9]),
       Vmax    = θ[11]*exp(η[10]),
       Km      = θ[12]*exp(η[11]))
end

θ = [
   1, # Ka1  Absorption rate constant 1 (1/time)
   1, # CL   Clearance (volume/time)
  20, # Vc   Central volume (volume)
   2, # Q    Inter-compartmental clearance (volume/time)
  10, # Vp   Peripheral volume of distribution (volume)
  10, # Kin  Response in rate constant (1/time)
   2, # Kout Response out rate constant (1/time)
   2, # IC50 Concentration for 50% of max inhibition (mass/volume)
   1, # IMAX Maximum inhibition
   1, # γ    Emax model sigmoidicity
   0, # Vmax Maximum reaction velocity (mass/time)
   2  # Km   Michaelis constant (mass/volume)
   ]

resp_0 = θ[6]/θ[7] # Kin/Kout
prob = ODEProblem(f,[0.0,0.0,0.0,resp_0],(0.0,240.0))
pkpd = PKPDModel(prob,set_parameters)
η = zeros(11)
sol  = simulate(pkpd,θ,η,data[1],abstol=1e-12,reltol=1e-12)

obsdata = process_data(joinpath(Pkg.dir("PKPDSimulator"),
           "examples/event_data","data24.csv"),Symbol[],Symbol[:ev1,:cp,:periph,:resp],
           separator=',')
obs_ev1s = obsdata.subjects[1].obs.vals[1]
obs_cps = obsdata.subjects[1].obs.vals[2]
obs_periphs = obsdata.subjects[1].obs.vals[3]
obs_resps = obsdata.subjects[1].obs.vals[4]

ev1s = sol(obs_times;idxs=1).u
@test maximum(ev1s - obs_ev1s) < 1e-6

cps = sol(obs_times;idxs=2)./θ[3]
@test maximum(cps - obs) < 1e-6

periphs = sol(obs_times;idxs=3).u
@test maximum(periphs - obs_periphs) < 1e-6

resps = sol(obs_times;idxs=4).u
@test maximum(resps - obs_resps) < 1e-6
