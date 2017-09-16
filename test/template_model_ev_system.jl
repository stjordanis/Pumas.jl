using PKPDSimulator


# Gut dosing model:
using ParameterizedFunctions
f = @ode_def_nohes GutDose begin
  dGut = -Ka*Gut
  dCent = Ka*Gut - (CL/V)*Cent
end Ka=>1.5 CL=>1.0 V=>30.0 LAGT=>0, MODE=>0, DUR2=>2, RAT2=>10, BIOAV=>1


function set_parameters!(p,u0,θ,η,zi)
  F_Cent =  BIOAV
  ALAG_CENT = LAGT
  if MODE==1
    R_CENT = RAT2
  if MODE==2
    D_CENT = DUR2
  Ka = θ[1]
  CL = θ[2]
  V  = θ[3]
  p[1] = Ka; p[2] = CL; p[3] = V
end


function getsol(model,tstart=0,tend=72,num_dv=1)
    tspan = (tsart,tend)
    num_dependent = num_dv
    sol  = simulate(f=model,tspan,num_dependent,set_parameters!,θ,ω,z)
    # todo: implement output function to derive concentrations at everytime point
    # by dividing the sol by the volume V
end

# ev1 - gut dose - use ev1.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg dose into gut compartment
# cmt=1: in the system of diffeq's, gut compartment is the first compartment
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
     1     #BIOAV
]

# corresponding mrgsolve and NONMEM solution in data1.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=2) # get both gut and central amounts and concentrations amt/V


# ev2 - infusion into the central compartment - use ev2.csv in PKPDSimulator/examples/event_data/

# new
# amt=100: 100 mg infusion into central compartment

# new
# cmt=2: in the system of diffeq's, central compartment is the second compartment

# new
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr

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
    1     #BIOAV
]

# corresponding mrgsolve and NONMEM solution in data2.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=1) # get central amounts  and concentrations amt/V


# ev3 - infusion into the central compartment with lag time - use ev3.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr

# new
# LAGT=5: there is a lag of 5 hours after dose administration when amounts from the event
# are populated into the central compartment. Requires developing a new internal variable called 
# ALAG_<comp name> or ALAG_<comp_num> that takes a time value by which the entry of dose into that compartment
# is delayed

# addl=3: 4 doses total, 1 dose at time zero + 3 additional doses (addl=3)
# ii=12: each additional dose is given with a frequency of ii=12 hours
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
    1     #BIOAV
]

# corresponding mrgsolve and NONMEM solution in data3.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=1) # get central amounts  and concentrations amt/V


# ev4 - infusion into the central compartment with lag time - use ev4.csv in PKPDSimulator/examples/event_data/
# amt=100: 100 mg infusion into central compartment
# cmt=2: in the system of diffeq's, central compartment is the second compartment
# rate=10: the dose is given at a rate of amt/time (mg/hr), i.e, 10mg/hr
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


θ = [ 
    1.5,  #Ka
    1.0,  #CL
    30.0, #V
    5,    #LAGT
    0,    #MODE
    2,    #DUR2
    10,   #RAT2
    0.412     #BIOAV
]

# corresponding mrgsolve and NONMEM solution in data4.csv in PKPDSimulator/examples/event_data/
sol = getsol(model=f,num_dv=1) # get central amounts  and concentrations amt/V