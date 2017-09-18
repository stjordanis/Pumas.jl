using PKPDSimulator, Base.Test, DataFrames

# Gut dosing model:
using ParameterizedFunctions
f = @ode_def_nohes GutDose begin
  dGut = -Ka*Gut
  dCent = Ka*Gut - (CL/V)*Cent
end Ka=>1.5 CL=>1.0 V=>30.0 #LAGT=>0, MODE=>0, DUR2=>2, RAT2=>10, BIOAV=>1

function set_parameters!(p,θ,η,zi)
    Ka = θ[1]
    CL = θ[2]*exp(η[1])
    V  = θ[3]*exp(η[2])
    p[1] = Ka; p[2] = CL; p[3] = V
end

function get_sol!(θ,z,obs,obs_times;
                       num_dv=2,cmt=2,kwargs...)
    tspan = (0.0,72.0)
    ω = zeros(2)
    num_dependent = num_dv
    sol  = simulate(f,tspan,num_dv,set_parameters!,θ,ω,z;kwargs...)
end

function get_residual!(θ,z,obs,obs_times;
                       num_dv=2,cmt=2,kwargs...)
    tspan = (0.0,72.0)
    ω = zeros(2)
    num_dependent = num_dv
    sol  = simulate(f,tspan,num_dv,set_parameters!,θ,ω,z;kwargs...)
    cps = sol[1](obs_times;idxs=cmt)./(θ[3]/1000)
    resid = cps - obs
end

function get_nonem_data(i)
    z = process_data(joinpath(Pkg.dir("PKPDSimulator"),
                  "examples/event_data/ev$i.csv"), covariates,dvs,
                  separator=',')
    raw_data = readtable(joinpath(Pkg.dir("PKPDSimulator"),
                "examples/event_data","data$i.csv"),
                separator=',')
    obs_idxs = find(x ->  x==0, raw_data[:evid])
    obs = raw_data[obs_idxs,:CP]
    obs_times = raw_data[obs_idxs,:time]
    z,obs,obs_times
end

# Fake dvs and covariates for testing
covariates = [1]
dvs = [1]

###############################
# Test 1
###############################

# ev1 - gut dose - use ev1.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg dose into gut compartment
# cmt=1: in the system of diffeq's, gut compartment is the first compartment
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=12: each additional dose is given with a frequency of ii=12 hours
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

z,obs,obs_times = get_nonem_data(1)

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

resid  = get_residual!(θ,z,obs,obs_times,
                abstol=1e-12,reltol=1e-12)
@test norm(resid) < 1e-3

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

z,obs,obs_times = get_nonem_data(2)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0 #V
    ]

sol  = get_sol!(θ,z,obs,obs_times,
                abstol=1e-12,reltol=1e-12)

resid  = get_residual!(θ,z,obs,obs_times,
                abstol=1e-12,reltol=1e-12)
norm(resid) < 1

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
z,obs,obs_times = get_nonem_data(3)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    5.0   #LAGT
    ]

resid  = get_residual!(θ,z,obs,obs_times,
                abstol=1e-12,reltol=1e-12)
norm(resid) < 1

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
z,obs,obs_times = get_nonem_data(4)

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    5.0   #LAGT
    ]

function set_parameters!(p,θ,η,zi)
    Ka = θ[1]
    CL = θ[2]*exp(η[1])
    V  = θ[3]*exp(η[2])
    ALAG_CENT = LAGT
    F_CENT = BIOAV
    p[1] = Ka; p[2] = CL; p[3] = V; p[4] = LAGT; p[5] = BIOAV
end

θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    5.0,  #LAGT
    0.412,#BIOAV
    ]

resid  = get_residual!(θ,z,obs,obs_times,
                abstol=1e-12,reltol=1e-12)
norm(resid) < 1

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


θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0,    #LAGT
    0,    #MODE
    2,    #DUR2
    10,   #RAT2
    0.412,#BIOAV
    1     #ss
    ]

# corresponding mrgsolve and NONMEM solution in data5.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=1) # get central amounts  and concentrations in central u_central/V

raw_data = readtable(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/data5.csv"),
              separator=',')

obs_idxs = find(x ->  x==0, raw_data[:evid])
obs = raw_data[obs_idxs,:CP]
obs_times = raw_data[obs_idxs,:time]
cps = sol[1](obs_times;idxs=2)./θ[3]

resid = 1000cps - obs # Why the scaling difference?
norm(resid) < 1

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


θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0,    #LAGT
    0,    #MODE
    2,    #DUR2
    10,   #RAT2
    0.812,#BIOAV
    1     #ss
    ]

# corresponding mrgsolve and NONMEM solution in data6.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=1) # get central amounts  and concentrations in central u_central/V

raw_data = readtable(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/data6.csv"),
              separator=',')

obs_idxs = find(x ->  x==0, raw_data[:evid])
obs = raw_data[obs_idxs,:CP]
obs_times = raw_data[obs_idxs,:time]
cps = sol[1](obs_times;idxs=2)./θ[3]

resid = 1000cps - obs # Why the scaling difference?
norm(resid) < 1

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


θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0,    #LAGT
    0,    #MODE
    2,    #DUR2
    10,   #RAT2
    1,    #BIOAV
    1     #ss
    ]

# corresponding mrgsolve and NONMEM solution in data7.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=1) # get central amounts  and concentrations in central u_central/V

raw_data = readtable(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/data7.csv"),
              separator=',')

obs_idxs = find(x ->  x==0, raw_data[:evid])
obs = raw_data[obs_idxs,:CP]
obs_times = raw_data[obs_idxs,:time]
cps = sol[1](obs_times;idxs=2)./θ[3]

resid = 1000cps - obs # Why the scaling difference?
norm(resid) < 1

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


θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0,    #LAGT
    0,    #MODE
    2,    #DUR2
    10,   #RAT2
    1,    #BIOAV
    1     #ss
    ]

# corresponding mrgsolve and NONMEM solution in data8.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=1) # get central amounts  and concentrations in central u_central/V

raw_data = readtable(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/data8.csv"),
              separator=',')

obs_idxs = find(x ->  x==0, raw_data[:evid])
obs = raw_data[obs_idxs,:CP]
obs_times = raw_data[obs_idxs,:time]
cps = sol[1](obs_times;idxs=2)./θ[3]

resid = 1000cps - obs # Why the scaling difference?
norm(resid) < 1

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
# only 81.2 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

#ss=1:  indicates that the dose is a steady state dose, and that the compartment amounts are to be reset
#to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior
#dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)

#new
# ii=10: each additional dose is given with a frequency of ii=10 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record


θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0,    #LAGT
    0,    #MODE
    2,    #DUR2
    10,   #RAT2
    0.412,#BIOAV
    1     #ss
    ]

# corresponding mrgsolve and NONMEM solution in data9.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=1) # get central amounts  and concentrations in central u_central/V

raw_data = readtable(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/data9.csv"),
              separator=',')

obs_idxs = find(x ->  x==0, raw_data[:evid])
obs = raw_data[obs_idxs,:CP]
obs_times = raw_data[obs_idxs,:time]
cps = sol[1](obs_times;idxs=2)./θ[3]

resid = 1000cps - obs # Why the scaling difference?
norm(resid) < 1

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
# only 81.2 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME

#ss=1:  indicates that the dose is a steady state dose, and that the compartment amounts are to be reset
#to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior
#dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=10: each additional dose is given with a frequency of ii=10 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record


θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0,    #LAGT
    0,    #MODE
    2,    #DUR2
    10,   #RAT2
    1,    #BIOAV
    1     #ss
    ]

# corresponding mrgsolve and NONMEM solution in data10.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=1) # get central amounts  and concentrations in central u_central/V

raw_data = readtable(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/data10.csv"),
              separator=',')

obs_idxs = find(x ->  x==0, raw_data[:evid])
obs = raw_data[obs_idxs,:CP]
obs_times = raw_data[obs_idxs,:time]
cps = sol[1](obs_times;idxs=2)./θ[3]

resid = 1000cps - obs # Why the scaling difference?
norm(resid) < 1

# ev11 - gut dose at steady state with lower bioavailability
# - use ev11.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment

#new
# cmt=1: in the system of diffeq's, central compartment is the second compartment

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
# ii=12: each additional dose is given with a frequency of ii=10 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record


θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    0,    #LAGT
    0,    #MODE
    2,    #DUR2
    10,   #RAT2
    0.412,#BIOAV
    1     #ss
    ]

# corresponding mrgsolve and NONMEM solution in data11.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=2) # get both gut and central amounts  and concentrations in central u_central/V

raw_data = readtable(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/data11.csv"),
              separator=',')

obs_idxs = find(x ->  x==0, raw_data[:evid])
obs = raw_data[obs_idxs,:CP]
obs_times = raw_data[obs_idxs,:time]
cps = sol[1](obs_times;idxs=2)./θ[3]

resid = 1000cps - obs # Why the scaling difference?
norm(resid) < 1

# ev12 - gut dose at with lower bioavailability and a 5 hour lag time
# - use ev12.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=1: in the system of diffeq's, central compartment is the second compartment
# BIOAV=0.412: required developing a new internal variable called F_<comp name> or F_<comp num>,
# where F is the fraction of amount that is delivered into the compartment. e.g. in this case,
# only 41.2 % of the 100 mg dose is administered at the 10mg/hr rate will enter the system.
# F_<comp> is one of the most commonly estimated parameters in NLME
# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)

#new
# LAGT=5: there is a lag of 5 hours after dose administration when amounts from the event
# are populated into the central compartment. Requires developing a new internal variable called
# ALAG_<comp name> or ALAG_<comp_num> that takes a value that delays the entry of dose into that compartment

# ii=12: each additional dose is given with a frequency of ii=10 hours

# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record


θ = [
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    5,    #LAGT
    0,    #MODE
    2,    #DUR2
    10,   #RAT2
    0.412 #BIOAV
    ]

# corresponding mrgsolve and NONMEM solution in data12.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=2) # get both gut and central amounts  and concentrations in central u_central/V

raw_data = readtable(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/data12.csv"),
              separator=',')

obs_idxs = find(x ->  x==0, raw_data[:evid])
obs = raw_data[obs_idxs,:CP]
obs_times = raw_data[obs_idxs,:time]
cps = sol[1](obs_times;idxs=2)./θ[3]

resid = 1000cps - obs # Why the scaling difference?
norm(resid) < 1
