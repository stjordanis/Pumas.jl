using PKPDSimulator, Base.Test, NamedTuples

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
function f(t,u,p,du)
 ev1,cent,periph,resp = u
 cp     = (cent/p.Vc)
 ct     = (periph/p.Vp)
 CLNL   = (p.Vmax/p.Km+cp)
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
function f(t,u,p,du)
    ev1,cent,periph,resp = u
    cp     = (cent/p.Vc)
    ct     = (periph/p.Vp)
    CLNL   = (p.Vmax/p.Km+cp)
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