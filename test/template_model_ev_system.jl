using PKPDSimulator, Base.Test, NamedTuples

# Gut dosing model
function f(t,u,p,du)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]))
end

function get_sol(θ,data;num_dv=2,kwargs...)
    prob = ODEProblem(f,zeros(num_dv),(0.0,72.0))
    pkpd = PKPDModel(prob,set_parameters)
    η = zeros(2)
    sol  = simulate(pkpd,θ,η,data[1];kwargs...)
end

function get_a_sol(θ,data;kwargs...)
   prob = OneCompartmentModel(72.0)
   pkpd = PKPDModel(prob,set_parameters)
   η = zeros(2)
   sol  = simulate(pkpd,θ,η,data[1];kwargs...)
end

function get_residual(θ,data,obs,obs_times;
                       num_dv=2,cmt=2,scaling_factor = 1000,kwargs...)
    sol = get_sol(θ,data;num_dv=num_dv,kwargs...)
    cps = sol(obs_times;idxs=cmt)./(θ[3]/scaling_factor)
    resid = cps - obs
end

function get_analytical_residual(θ,data,obs,obs_times;scaling_factor = 1000,kwargs...)
    sol = get_a_sol(θ,data;kwargs...)
    cps = sol(obs_times;idxs=2)./(θ[3]/scaling_factor)
    resid = cps - obs
end

function get_nonem_data(i)
    data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
                  "examples/event_data/ev$i.csv"), covariates,dvs,
                  separator=',')
    obsdata = process_data(joinpath(Pkg.dir("PKPDSimulator"),
                "examples/event_data","data$i.csv"),Symbol[],Symbol[:cp],
                separator=',')
    obs = map(x -> x.cp, obsdata.patients[1].obs)
    obs_times = obsdata.patients[1].obs_times
    data,obs,obs_times
end

# Fake dvs and covariates for testing
covariates = [1]
dvs = [1]

###############################
# Test 2
###############################

# ev2 - infusion into the central compartment - use ev2.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment

# new
# cmt=2: in the system of diffeq's, central compartment is the second compartment

# new
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr.
# In this example the 100mg amount is given over a duration (DUR) of 10 hours

# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=12: each additional dose is given with a frequency of ii=12 hours
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(2)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0 #V
    ]

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)

@test norm(resid) < 1e-6

sol  = get_sol(θ,data)
asol  = get_a_sol(θ,data)

a_resid  = get_analytical_residual(θ,data,obs,obs_times)

@test norm(a_resid) < 1e-6

###############################
# Test 3
###############################

# ev3 - infusion into the central compartment with lag time
# - use ev3.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr. In this example the 100mg amount
# is given over a duration (DUR) of 10 hours

# new
# LAGT=5: there is a lag of 5 hours after dose administration when amounts from the event
# are populated into the central compartment. Requires developing a new internal variable called
# ALAG_<comp name> or ALAG_<comp_num> that takes a time value by which the entry of dose into that compartment
# is delayed

# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=12: each additional dose is given with a frequency of ii=12 hours
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record
data,obs,obs_times = get_nonem_data(3)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    5.0   #LAGT
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        lags = θ[4])
end

sol  = get_sol(θ,data,abstol=1e-12,reltol=1e-12)

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-6

a_resid  = get_analytical_residual(θ,data,obs,obs_times)
@test norm(a_resid) < 1e-6

###############################
# Test 4
###############################

# ev4 - infusion into the central compartment with lag time and bioavailability
# - use ev4.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr. In this example the 100mg amount
# is given over a duration (DUR) of 10 hours
# LAGT=5: there is a lag of 5 hours after dose administration when amounts from the event
# are populated into the central compartment. Requires developing a new internal variable called
# ALAG_<comp name> or ALAG_<comp_num> that takes a value that delays the entry of dose into that compartment

#new
# BIOAV=0.412: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# only 41.2 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=12: each additional dose is given with a frequency of ii=12 hours
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record
data,obs,obs_times = get_nonem_data(4)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    5.0,  #LAGT
    0.412,#BIOAV
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        lags = θ[4],
        bioav = θ[5])
end

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-6

# Make sure modifications are handled correctly with base_time by repeating
resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-6

a_resid  = get_analytical_residual(θ,data,obs,obs_times)
@test norm(a_resid) < 1e-6

###############################
# Test 5
###############################

# ev5 - infusion into the central compartment at steady state (ss)
# - use ev5.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr. In this example the 100mg amount
# is given over a duration (DUR) of 10 hours
# BIOAV=0.412: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# only 41.2 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

#new
#ss=1:  indicates that the dose is a steady state dose, and that the compartment amounts are to be reset
#to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior
#dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled

# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=12: each additional dose is given with a frequency of ii=12 hours
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(5)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0.412,#BIOAV
    10,   #RAT2
    1     #ss
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        bioav = θ[4])
end

function analytical_ss_update(u0,rate,duration,deg,bioav,ii)
    rate_on_duration = duration*bioav
    rate_off_duration = ii-rate_on_duration
    ee = exp(deg*rate_on_duration)
    u_rate_off = inv(ee)*(-rate + ee*rate + deg*u0)/deg
    u = exp(-deg*rate_off_duration)*u_rate_off
    u
end

u0 = 0.0
for i in 1:200
    u0 = analytical_ss_update(u0,10,10,θ[2]/θ[3],θ[4],12)
end

sol  = get_sol(θ,data,abstol=1e-14,reltol=1e-14)
@test norm(sol[3][2] - u0) < 1e-9

asol  = get_a_sol(θ,data,abstol=1e-14,reltol=1e-14)
@test norm(asol[1][2] - u0) < 1e-9

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-14,reltol=1e-14)
@test norm(resid) < 1e-2

a_resid  = get_analytical_residual(θ,data,obs,obs_times)[2:end]
@test norm(a_resid) < 1e-2

###############################
# Test 6
###############################

# ev6 - infusion into the central compartment at steady state (ss), where frequency of events (ii) is less
# than the infusion duration (DUR)
# - use ev6.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr. In this example the 100mg amount
# is given over a duration (DUR) of 10 hours

#new
# BIOAV=0.812: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# only 81.2 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

#ss=1:  indicates that the dose is a steady state dose, and that the compartment amounts are to be reset
#to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior
#dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)

#new
# ii=6: each additional dose is given with a frequency of ii=6 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(6)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0.812,#BIOAV
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        bioav = θ[4])
end

function analytical_ss_update(u0,rate,duration,deg,bioav,ii)
    rate_on_duration = duration*bioav - ii
    rate_off_duration = ii - rate_on_duration
    ee = exp(deg*rate_on_duration)
    u_rate_off = inv(ee)*(-2rate + ee*2rate + deg*u0)/deg
    ee2 = exp(deg*rate_off_duration)
    u = inv(ee2)*(-rate + ee2*rate + deg*u_rate_off)/deg
    u
end

sol  = get_sol(θ,data,abstol=1e-12,reltol=1e-12)
asol  = get_a_sol(θ,data,abstol=1e-12,reltol=1e-12)
u0 = 0.0
for i in 1:200
    u0 = analytical_ss_update(u0,10,10,θ[2]/θ[3],θ[4],6)
end
@test norm(sol[3][2]  - u0) < 1e-9
@test norm(asol[1][2] - u0) < 1e-9

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-2

a_resid  = get_analytical_residual(θ,data,obs,obs_times)[2:end]
@test norm(a_resid) < 1e-2

###############################
# Test 7
###############################

# ev7 - infusion into the central compartment at steady state (ss), where frequency of events (ii) is less
# than the infusion duration (DUR)
# - use ev7.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr. In this example the 100mg amount
# is given over a duration (DUR) of 10 hours

#new
# BIOAV=1: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# 100 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

#ss=1:  indicates that the dose is a steady state dose, and that the compartment amounts are to be reset
#to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior
#dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)

#new
# ii=6: each additional dose is given with a frequency of ii=6 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(7)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    1,    #BIOAV
    10,   #RAT2
    1     #ss
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        bioav = θ[4])
end

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-2

a_resid  = get_analytical_residual(θ,data,obs,obs_times)[2:end]
@test norm(a_resid) < 1e-2


###############################
# Test 8
###############################

# ev8 - infusion into the central compartment at steady state (ss), where frequency of events (ii) is a
# multiple of infusion duration (DUR)
# - use ev8.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment

#new
# rate=8.33333: the dose is given at a rate of amt/time (mg/hr), i.e, 8.333333mg/hr. In this example the 100mg amount
# is given over a duration (DUR) of 12 hours


# BIOAV=1: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# 100 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

#ss=1:  indicates that the dose is a steady state dose, and that the compartment amounts are to be reset
#to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior
#dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)


# ii=6: each additional dose is given with a frequency of ii=6 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(8)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    1,    #BIOAV
    10,   #RAT2
    1     #ss
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        bioav = θ[4])
end

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-2

a_resid  = get_analytical_residual(θ,data,obs,obs_times)[2:end]
@test norm(a_resid) < 1e-2


###############################
# Test 9
###############################

# ev9 - infusion into the central compartment at steady state (ss), where frequency of events (ii) is
# exactly equal to infusion duration (DUR)
# - use ev9.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment

#new
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr. In this example the 100mg amount
# is given over a duration (DUR) of 10 hours

#new
# BIOAV=0.412: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# only 41.2 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

#ss=1:  indicates that the dose is a steady state dose, and that the compartment amounts are to be reset
#to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior
#dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)

#new
# ii=10: each additional dose is given with a frequency of ii=10 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(9)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0.412,#BIOAV
    10,   #RAT2
    1     #ss
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        bioav = θ[4])
end

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-2

a_resid  = get_analytical_residual(θ,data,obs,obs_times)[2:end]
@test norm(a_resid) < 1e-2


###############################
# Test 10
###############################

# ev10 - infusion into the central compartment at steady state (ss), where frequency of events (ii) is
# exactly equal to infusion duration (DUR)
# - use ev10.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr. In this example the 100mg amount
# is given over a duration (DUR) of 10 hours

#new
# BIOAV=1: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# 100 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

#ss=1:  indicates that the dose is a steady state dose, and that the compartment amounts are to be reset
#to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior
#dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=10: each additional dose is given with a frequency of ii=10 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(10)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    1,    #BIOAV
    10,   #RAT2
    1     #ss
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        bioav = θ[4])
end

sol  = get_sol(θ,data,abstol=1e-12,reltol=1e-12)
asol  = get_a_sol(θ,data,abstol=1e-12,reltol=1e-12)

res = 1000sol(obs_times;idxs=2)/30

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-2

a_resid  = get_analytical_residual(θ,data,obs,obs_times)[2:end]
@test norm(a_resid) < 1e-2


###############################
# Test 11
###############################

# ev11 - gut dose at steady state with lower bioavailability
# - use ev11.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg bolus into depot compartment

#new
# cmt=1: in the system of diffeq's, gut compartment is the first compartment

#new
# BIOAV=0.412: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# only 41.2 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

#ss=1:  indicates that the dose is a steady state dose, and that the compartment amounts are to be reset
#to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior
#dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)

#new
# ii=12: each additional dose is given with a frequency of ii=12 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(11)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    1.0, #BIOAV
    10,   #RAT2
    1     #ss
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        bioav = θ[4])
end

sol  = get_sol(θ,data,abstol=1e-12,reltol=1e-12)
asol  = get_a_sol(θ,data,abstol=1e-12,reltol=1e-12)

analytical_f = OneCompartmentModel(0.0).f
p = @NT(Ka = θ[1],
        CL = θ[2],
        V  = θ[3])
u0 = zeros(2)
for i in 1:200
    u0 = convert(Array,analytical_f(12,0.0,u0,θ[4]*[100.0,0.0],p,zeros(2)))
end
u0[1] += θ[4]*100.0

@test norm(sol[3] - u0) < 1e-9

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-2

a_resid  = get_analytical_residual(θ,data,obs,obs_times)[2:end]
@test norm(a_resid) < 1e-2

###############################
# Test 12
###############################

# ev12 - gut dose at with lower bioavailability and a 5 hour lag time
# - use ev12.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg bolus into gut compartment
# cmt=1: in the system of diffeq's, gut compartment is the first compartment
# BIOAV=0.412: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# only 41.2 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)

#new
# LAGT=5: there is a lag of 5 hours after dose administration when amounts from the event
# are populated into the central compartment. Requires developing a new internal variable called
# ALAG_<comp name> or ALAG_<comp_num> that takes a value that delays the entry of dose into that compartment

# ii=12: each additional dose is given with a frequency of ii=12 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(12)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0,  #V
    5.0,  #LAGT
    0.412 #BIOAV
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]))
        #lag = θ[4],
        #bioav = θ[5]
end

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-6

a_resid  = get_analytical_residual(θ,data,obs,obs_times)
@test norm(a_resid) < 1e-6

###############################
# Test 13
###############################

# ev13 - zero order infusion followed by first order absorption into gut
# - use ev13.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into gut compartment at time zero
# amt=50; 50 mg bolus into gut compartment at time = 12 hours
# cmt=1: in the system of diffeq's, gut compartment is the first compartment

#new
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr. In this example the 100mg amount
# is given over a duration (DUR) of 10 hours
# BIOAV=1: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# 100 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(13)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0,  #V
    1 #BIOAV
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        bioav = θ[4])
end

sol  = get_sol(θ,data,abstol=1e-12,reltol=1e-12)
asol  = get_a_sol(θ,data,abstol=1e-12,reltol=1e-12)

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-6

a_resid  = get_analytical_residual(θ,data,obs,obs_times)
@test norm(a_resid) < 1e-6

###############################
# Test 14
###############################

# ev14 - zero order infusion into central compartment specified by duration parameter
# - use ev14.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment at time zero

#new
# cmt=2: in the system of diffeq's, central compartment is the second compartment


# rate= - 2 : when a dataset specifies rate = -2 in an event row, then infusions are modeled via the duration parameter

# DUR2 = drug is adminstered over a 9 hour duration into the central compartment

# LAGT=5: there is a lag of 5 hours after dose administration when amounts from the event
# are populated into the central compartment. Requires developing a new internal variable called
# ALAG_<comp name> or ALAG_<comp_num> that takes a value that delays the entry of dose into that compartment

# BIOAV=0.61: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# 61 % of the 100 mg dose is administered over 9 hours duration.
# F_<comp> is one of the most commonly estimated parameters in NLME

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(14)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0,  #V
    0.61, #BIOAV
    5.0, #LAGT
    9.0  #duration
    ]

function set_parameters(θ,η,z)
    @NT(Ka = θ[1],
        CL = θ[2]*exp(η[1]),
        V  = θ[3]*exp(η[2]),
        bioav = θ[4],
        lags = θ[5],
        duration = θ[6])
end

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-6

a_resid  = get_analytical_residual(θ,data,obs,obs_times)
@test norm(a_resid) < 1e-6

###############################
# Test 15
###############################

## SS=2 and next dose overlapping into the SS interval
# ev15 - first order bolus into central compartment at ss followed by an ss=2 (superposition ss) dose at 12 hours
# - use ev15.csv in PKPDSimulator/examples/event_data/
# amt=10: 10 mg bolus into central compartment at time zero using ss=1, followed by a 20 mg ss=2 dose at time 12
# cmt=2: in the system of diffeq's, central compartment is the second compartment

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(15)

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

sol  = get_sol(θ,data,abstol=1e-12,reltol=1e-12)
asol  = get_a_sol(θ,data,abstol=1e-12,reltol=1e-12)

# Use post-dose observations
resid = sol(obs_times[1:end-19]+1e-12;idxs=2)/θ[3] - obs[1:end-19]
@test norm(resid) < 1e-5
resid = asol(obs_times[1:end-19]+1e-12;idxs=2)/θ[3] - obs[1:end-19]
@test_broken norm(resid) < 1e-5

#@test_broken resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
#@test_broken norm(resid) < 1e-6

#a_resid = get_analytical_residual(θ,data,obs,obs_times)
#@test_broken norm(a_resid) < 1e-7

###############################
# Test 16
###############################

## SS=2 with a no-reset afterwards
# ev16 - first order bolus into central compartment at ss followed by
# an ss=2 (superposition ss) dose at 12 hours followed by reset ss=1 dose at 24 hours
# - use ev16.csv in PKPDSimulator/examples/event_data/
# amt=10: 10 mg bolus into central compartment at time zero using ss=1, followed by 20 mg ss=2 dose at time 12 followed
# 10 mg ss = 1 reset dose at time 24
# cmt=2: in the system of diffeq's, central compartment is the second compartment

#new
# BIOAV=1: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# 100 % of the 100 mg dose is administered over 9 hours duration.
# F_<comp> is one of the most commonly estimated parameters in NLME

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(16)

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

sol  = get_sol(θ,data,abstol=1e-12,reltol=1e-12)
asol  = get_a_sol(θ,data,abstol=1e-12,reltol=1e-12)

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-4

a_resid = get_analytical_residual(θ,data,obs,obs_times)[2:end]
@test_broken norm(a_resid) < 1e-7


###############################
# Test 17
###############################

# ev2_const_infusion.csv - zero order constant infusion at time=10 followed by infusion at time 15
# - use ev17.csv in PKPDSimulator/examples/event_data/
# several observations predose (time<10) even though time=10 is a constant infusion as steady state (SS=1)
# amt=0: constant infusion with rate=10 at time 10
# amt=200; 200 dose units infusion with rate=20 starting at time 15
# cmt=2: doses in the central compartment in a first order absorption model
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(17)

θ = [
    1.0,  #Ka
    1.0,  #CL
    30.0 #V
    ]

function set_parameters(θ,η,z)
   @NT(Ka = θ[1],
    CL = θ[2]*exp(η[1]),
    V  = θ[3]*exp(η[2]))
end

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12,scaling_factor=1)
@test norm(resid) < 1e-6

@test_broken a_resid = get_analytical_residual(θ,data,obs,obs_times)
@test_broken norm(a_resid) < 1e-7

###############################
# Test 18
###############################

# ev2_const_infusion2.csv - zero order constant infusion at all observations
# - use ev18.csv in PKPDSimulator/examples/event_data/
# several constant infusion dose rows (SS=1, amt=0, rate=10) are added previous to each observation
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(18)

θ = [
    1.0,  #Ka
    1.0,  #CL
    30.0 #V
    ]

function set_parameters(θ,η,z)
   @NT(Ka = θ[1],
    CL = θ[2]*exp(η[1]),
    V  = θ[3]*exp(η[2]))
end

sol  = get_sol(θ,data,abstol=1e-12,reltol=1e-12)
resid = sol(obs_times+1e-14;idxs=2)/θ[3] - obs
# use post-dose observations
#resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12,scaling_factor=1)
@test norm(resid) < 1e-6

@test_broken a_resid = get_analytical_residual(θ,data,obs,obs_times)
@test_broken norm(a_resid) < 1e-7

###############################
# Test 19
###############################

# ev19 - Two parallel first order absorption models
# use ev19.csv in PKPDSimulator/examples/event_data/
# In some cases, after oral administration, the plasma concentrations exhibit a double
# peak or shouldering-type absorption.
# gut compartment is split into two compartments Depot1 and Depot2
# a 10 mg dose is given into each of the gut compartments
# Depot1 has a bioav of 0.5 (50 %) and Depot2 has a bioav of 1 - 0.5 = 0.5 (note bioav should add up to 1)
# cmt=1: in the system of diffeq's, Depot1 compartment is the first compartment
# cmt=2: in the system of diffeq's, Depot2 compartment is the second compartment
# cmt=3: in the system of diffeq's, central compartment is the third compartment
# Depot2Lag = 5; a 5 hour lag before which the drug shows up from the depot2 compartment with a specified bioav
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(19)

# Parallel first order absorption dosing model
function f(t,u,p,du)
    Depot1, Depot2, Central = u
    du[1] = -p.Ka1*Depot1
    du[2] = -p.Ka2*Depot2
    du[3] =  p.Ka1*Depot1 + p.Ka2*Depot2 - (p.CL/p.V)*Central
   end

function set_parameters(θ,η,z)
   @NT(Ka1 = θ[1],
       Ka2 = θ[2],
       CL = θ[4]*exp(η[1]),
       V  = θ[3]*exp(η[2]),
       bioav = (θ[5],1 - θ[5],1),
       lags = (0,θ[6],0))
end

θ = [
     0.8,  #Ka1
     0.6,  #Ka2
     50.0, #V # V needs to be 3 for the test to scale the result properly
     5.0,  #CL
     0.5,  #bioav1
     5     #lag2
     ]

resid  = get_residual(θ,data,obs,obs_times,num_dv=3,cmt=3,abstol=1e-12,reltol=1e-12,scaling_factor=1)
@test norm(resid) < 1e-6

@test_broken a_resid = get_analytical_residual(θ,data,obs,obs_times,num_dv=3,cmt=3,scaling_factor=1)
@test_broken norm(a_resid) < 1e-7


###############################
# Test 20
###############################

# ev20 - Mixed zero and first order absorption
# use ev20.csv in PKPDSimulator/examples/event_data/
# For the current example, the first-order process starts immediately after dosing into the Depot (gut)
# and is followed, with a lag time (lag2), by a zero-order process in the central compartment.
# a 10 mg dose is given into the gut compartment (cmt=1) at time zero with a bioav of 0.5 (bioav1)
# Also at time zero a zero order dose with a 4 hour duration is given into the central compartment with a bioav2 of 1-bioav1 = 0.5
# Depot2Lag = 5; a 5 hour lag before which the drug shows up from the zero order process into the central compartment with the specified bioav2
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

data,obs,obs_times = get_nonem_data(20)

# Parallel first order absorption dosing model
function f(t,u,p,du)
    Depot, Central = u
    du[1] = -p.Ka*Depot
    du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
   end

function set_parameters(θ,η,z)
   @NT(Ka = θ[1],
       CL = θ[2]*exp(η[1]),
       V  = θ[3]*exp(η[2]),
       bioav = (θ[5],1 - θ[5]),
       duration = (0.0,4.0),
       lags = (0.0,θ[4]))
end

θ = [
     0.5,  #Ka1
     5.0,  #CL
     50.0, #V
     5,    #lag2
     0.5   #bioav1
     ]

resid  = get_residual(θ,data,obs,obs_times,abstol=1e-12,reltol=1e-12,scaling_factor=1)
@test norm(resid) < 1e-6

a_resid = get_analytical_residual(θ,data,obs,obs_times,scaling_factor=1)
@test norm(a_resid) < 1e-7
