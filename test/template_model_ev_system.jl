using Base.Test
using PKPDSimulator, NamedTuples, Distributions, PDMats


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

# Gut dosing model
m_diffeq = @model begin
    @param   θ ∈ VectorDomain(3, lower=zeros(3), init=ones(3))
    @random  η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
    end

    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    @post cp = Central / V
end

m_analytic = @model begin
    @param   θ ∈ VectorDomain(3, lower=zeros(3), init=ones(3))
    @random  η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
    end

    @dynamics OneCompartmentModel

    @post cp = Central / V
end

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data2.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [1.5,  #Ka
              1.0,  #CL
              30.0 #V
              ])
y0 = @NT(η = [0.0,0.0])

sim = pkpd_post(m_diffeq, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations]

sim = pkpd_post(m_analytic, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations]

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

mlag_diffeq = @model begin
    @param    θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
    @random   η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        lags = θ[4]
    end

    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    @post cp = Central / V
end

mlag_analytic = @model begin
    @param    θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
    @random   η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        lags = θ[4]
    end

    @dynamics OneCompartmentModel

    @post cp = Central / V
end

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data3.csv"),
                       [], [:cp],  separator=',')[1]


x0 = @NT(θ = [1.5,  #Ka
              1.0,  #CL
              30.0, #V
              5.0   #lags
              ])

sim = pkpd_post(mlag_diffeq, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations]

sim = pkpd_post(mlag_analytic, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations]


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

mlagbioav_diffeq = @model begin
    @param    θ ∈ VectorDomain(5, lower=zeros(5), init=ones(5))
    @random   η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        lags = θ[4]
        bioav = θ[5]
    end

    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    @post cp = Central / V
end

mlagbioav_analytic = @model begin
    @param    θ ∈ VectorDomain(5, lower=zeros(5), init=ones(5))
    @random   η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        lags = θ[4]
        bioav = θ[5]
    end

    @dynamics OneCompartmentModel

    @post cp = Central / V
end

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data4.csv"),
                       [], [:cp],  separator=',')[1]


x0 = @NT(θ = [1.5,  #Ka
              1.0,  #CL
              30.0, #V
              5.0,  #lags
              0.412,#bioav
              ])

sim = pkpd_post(mlagbioav_diffeq, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations]

sim = pkpd_post(mlagbioav_analytic, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations]


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

mbioav_diffeq = @model begin
    @param    θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
    @random   η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        bioav = θ[4]
    end

    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    @post cp = Central / V
end

mbioav_analytic = @model begin
    @param    θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
    @random   η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        bioav = θ[4]
    end

    @dynamics OneCompartmentModel

    @post cp = Central / V
end

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data5.csv"),
                       [], [:cp],  separator=',')[1]


x0 = @NT(θ = [1.5,  #Ka
              1.0,  #CL
              30.0, #V
              0.412,#bioav
              ])

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
    u0 = analytical_ss_update(u0,10,10,x0.θ[2]/x0.θ[3],x0.θ[4],12)
end

sol,col = pkpd_solve(mbioav_diffeq, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test sol[3][2] ≈ u0

sol,col = pkpd_solve(mbioav_analytic, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test sol[3][2] ≈ u0


sim = pkpd_post(mbioav_diffeq, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-5

sim = pkpd_post(mbioav_analytic, subject, x0, y0; abstol=1e-14, reltol=1e-14)
# TODO: why is the first value wrong?
@test [1000*v.cp for v in sim[2:end]] ≈ [obs.val.cp for obs in subject.observations[2:end]] rtol=1e-6


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data6.csv"),
                       [], [:cp],  separator=',')[1]


x0 = @NT(θ = [1.5,  #Ka
              1.0,  #CL
              30.0, #V
              0.812,#bioav
              ])

function analytical_ss_update(u0,rate,duration,deg,bioav,ii)
    rate_on_duration = duration*bioav - ii
    rate_off_duration = ii - rate_on_duration
    ee = exp(deg*rate_on_duration)
    u_rate_off = inv(ee)*(-2rate + ee*2rate + deg*u0)/deg
    ee2 = exp(deg*rate_off_duration)
    u = inv(ee2)*(-rate + ee2*rate + deg*u_rate_off)/deg
    u
end

u0 = 0.0
for i in 1:200
    u0 = analytical_ss_update(u0,10,10,x0.θ[2]/x0.θ[3],x0.θ[4],6)
end

sol,col = pkpd_solve(mbioav_diffeq, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test sol[3][2] ≈ u0

sol,col = pkpd_solve(mbioav_analytic, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test sol[3][2] ≈ u0


sim = pkpd_post(mbioav_diffeq, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-5

sim = pkpd_post(mbioav_analytic, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim[2:end]] ≈ [obs.val.cp for obs in subject.observations[2:end]] rtol=1e-6
# TODO: why is the first value wrong?


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data7.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [ 1.5,  #Ka
               1.0,  #CL
               30.0, #V
               1,    #BIOAV
               ])

sim = pkpd_post(mbioav_diffeq, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(mbioav_analytic, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim[2:end]] ≈ [obs.val.cp for obs in subject.observations[2:end]] rtol=1e-6
# TODO: why is the first value wrong?


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data8.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [ 1.5,  #Ka
               1.0,  #CL
               30.0, #V
               1,    #BIOAV
               ])

sim = pkpd_post(mbioav_diffeq, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(mbioav_analytic, subject, x0, y0; abstol=1e-14, reltol=1e-14)
@test [1000*v.cp for v in sim[2:end]] ≈ [obs.val.cp for obs in subject.observations[2:end]] rtol=1e-6
# TODO: why is the first value wrong?


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data9.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [ 1.5,  #Ka
               1.0,  #CL
               30.0, #V
               0.412,#BIOAV
               ])

sim = pkpd_post(mbioav_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
# TODO: this is wrong
#@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-5

sim = pkpd_post(mbioav_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim[2:end]] ≈ [obs.val.cp for obs in subject.observations[2:end]] rtol=1e-6
# TODO: why is the first value wrong?


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data10.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [ 1.5,  #Ka
               1.0,  #CL
               30.0, #V
               1,    #BIOAV
               ])

sim = pkpd_post(mbioav_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
# TODO: what is going on here?
#@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(mbioav_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim[2:end]] ≈ [obs.val.cp for obs in subject.observations[2:end]] rtol=1e-6
# TODO: why is the first value wrong?


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data11.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [ 1.5,  #Ka
               1.0,  #CL
               30.0, #V
               1.0, #BIOAV
               ])

sim = pkpd_post(mbioav_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(mbioav_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim[2:end]] ≈ [obs.val.cp for obs in subject.observations[2:end]] rtol=1e-6
# TODO: why is the first value wrong?


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data12.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [ 1.5,  #Ka
               1.0,  #CL
               30.0, #V
               ])

sim = pkpd_post(m_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(m_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data13.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [ 1.5,  #Ka
               1.0,  #CL
               30.0, #V
               1.0, #BIOAV
               ])

sim = pkpd_post(mbioav_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(mbioav_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6


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

mbld_diffeq = @model begin
    @param    θ ∈ VectorDomain(5, lower=zeros(5), init=ones(5))
    @random   η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        bioav = θ[4]
        lags = θ[5]
        duration = θ[6]
    end

    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    @post cp = Central / V
end

mbld_analytic = @model begin
    @param    θ ∈ VectorDomain(5, lower=zeros(5), init=ones(5))
    @random   η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        bioav = θ[4]
        lags = θ[5]
        duration = θ[6]
    end

    @dynamics OneCompartmentModel

    @post cp = Central / V
end

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data14.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [
              1.5,  #Ka
              1.0,  #CL
              30.0,  #V
              0.61, #BIOAV
              5.0, #LAGT
              9.0  #duration
              ])

sim = pkpd_post(mbld_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(mbld_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data15.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [
              1.5,  #Ka
              1.0,  #CL
              30.0 #V
              ])

# TODO: this is broken
sim = pkpd_post(m_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
# @test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

# Also, for some reason this is unscaled?
sim = pkpd_post(m_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data16.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [
              1.5,  #Ka
              1.0,  #CL
              30.0 #V
              ])

# TODO: this is broken
sim = pkpd_post(m_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
#@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

# TODO: wrong at dose times?
sim = pkpd_post(m_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
#@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6


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

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data17.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [
              1.0,  #Ka
              1.0,  #CL
              30.0 #V
              ])

sim = pkpd_post(m_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
# TODO: wrong at 12th obs?
#@test [v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(m_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6


###############################
# Test 18
###############################

# ev2_const_infusion2.csv - zero order constant infusion at all observations
# - use ev18.csv in PKPDSimulator/examples/event_data/
# several constant infusion dose rows (SS=1, amt=0, rate=10) are added previous to each observation
# evid = 1: indicates a dosing event
# mdv = 1: indicates that observations are not avaialable at this dosing record

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data18.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [
              1.0,  #Ka
              1.0,  #CL
              30.0  #V
              ])

# TODO: very wrong
sim = pkpd_post(m_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
#@test [v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(m_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6


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

mparbl_diffeq = @model begin
    @param   θ ∈ VectorDomain(6, lower=zeros(6), init=ones(6))
    @random  η ~ MvNormal(eye(2))

    @collate begin
        Ka1 = θ[1]
        Ka2 = θ[2]
        CL = θ[4]*exp(η[1])
        V  = θ[3]*exp(η[2])
        bioav = (θ[5],1 - θ[5],1)
        lags = (0,θ[6],0)
    end

    @dynamics begin
        dDepot1  = -Ka1*Depot1
        dDepot2  = -Ka2*Depot2
        dCentral =  Ka1*Depot1 + Ka2*Depot2 - (CL/V)*Central
    end

    @post cp = Central / V
end

mparbl_analytic = @model begin
    @param   θ ∈ VectorDomain(6, lower=zeros(6), init=ones(6))
    @random  η ~ MvNormal(eye(2))

    @collate begin
        Ka1 = θ[1]
        Ka2 = θ[2]
        CL = θ[4]*exp(η[1])
        V  = θ[3]*exp(η[2])
        bioav = (θ[5],1 - θ[5],1)
        lags = (0,θ[6],0)
    end

    @dynamics OneCompartmentParallelModel

    @post cp = Central / V
end

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data19.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [
              0.8,  #Ka1
              0.6,  #Ka2
              50.0, #V # V needs to be 3 for the test to scale the result properly
              5.0,  #CL
              0.5,  #bioav1
              5     #lag2
              ])


sim = pkpd_post(mparbl_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(mparbl_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6


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

mbl2_diffeq = @model begin
    @param   θ ∈ VectorDomain(5, lower=zeros(5), init=ones(5))
    @random  η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        bioav = (θ[5],1 - θ[5])
        duration = (0.0,4.0)
        lags = (0.0,θ[4])
    end

    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    @post cp = Central / V
end

mbl2_analytic = @model begin
    @param   θ ∈ VectorDomain(5, lower=zeros(5), init=ones(5))
    @random  η ~ MvNormal(eye(2))

    @collate begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
        bioav = (θ[5],1 - θ[5])
        duration = (0.0,4.0)
        lags = (0.0,θ[4])
    end

    @dynamics OneCompartmentModel

    @post cp = Central / V
end


subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data20.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [
              0.5,  #Ka1
              5.0,  #CL
              50.0, #V
              5,    #lag2
              0.5   #bioav1
              ])


sim = pkpd_post(mbl2_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(mbl2_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6


###############################
# Test 21
###############################

# ev21 - Testing evid=4
# use ev21.csv in PKPDSimulator/examples/event_data/
# For the current example, the first-order process starts immediately after dosing into the Depot (gut)
# at time=0 and evid=1 followed by a second dose into Depot at time=12 hours, but with evid=4
# A  10 mg dose is given into the gut compartment (cmt=1) at time zero with a bioav of 1 (bioav1)
# A second dose at time 12 hours is given into the gut but with evid=4 which should clear anything remaining in all compartments
# and give this dose.
# evid = 1: indicates a dosing event
# evid = 4: indicates a dosing event where time and amounts in all compartments are reset to zero
# mdv = 1: indicates that observations are not avaialable at this dosing record

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data21.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [
              1.5,  #Ka
              1.0,  #CL
              30.0  #V
              ])

sim = pkpd_post(m_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(m_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
# TODO: post-dose mixup at 13th observation?
inds = [1:12...,14:25...]
@test [1000*v.cp for v in sim][inds] ≈ [obs.val.cp for obs in subject.observations][inds] rtol=1e-6


###############################
# Test 22
###############################

# ev22 - Testing evid=4
# use ev22.csv in PKPDSimulator/examples/event_data/
# For the current example, a bolus dose is given into the central compartment at time=0 followed by a
# second dose into the gut compartment at time=12 with evid=4

# A  10 mg dose is given into the central compartment (cmt=2) at time zero with a bioav of 1 (bioav1)
# A second dose at time 12 hours is given into the gut but with evid=4 which should clear anything remaining in all compartments
# and give this dose.
# evid = 1: indicates a dosing event
# evid = 4: indicates a dosing event where time and amounts in all compartments are reset to zero
# mdv = 1: indicates that observations are not avaialable at this dosing record

subject = process_data(Pkg.dir("PKPDSimulator", "examples/event_data","data22.csv"),
                       [], [:cp],  separator=',')[1]

x0 = @NT(θ = [
              1.5,  #Ka
              1.0,  #CL
              30.0  #V
              ])

sim = pkpd_post(m_diffeq, subject, x0, y0; abstol=1e-12, reltol=1e-12)
@test [1000*v.cp for v in sim] ≈ [obs.val.cp for obs in subject.observations] rtol=1e-6

sim = pkpd_post(m_analytic, subject, x0, y0; abstol=1e-12, reltol=1e-12)
# TODO: post-dose mixup at 1st & 13th observation?
inds = [2:12...,14:25...]
@test [1000*v.cp for v in sim][inds] ≈ [obs.val.cp for obs in subject.observations][inds] rtol=1e-6
