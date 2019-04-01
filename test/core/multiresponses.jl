using PuMaS, Test, LinearAlgebra


###############################
# Test 23
###############################

# ev23 - Testing irm1
# - Indirect response model, type 1
# - Inhibition of response input
# - Two-compartment PK model
# - Optional nonlinear clearance - not being used in this example (Vmax and Km)
# use ev23.csv in PuMaS/examples/event_data/
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

subject = process_nmtran(example_nmtran_data("event_data/data23"),
                         [], [:ev1,:cp,:periph,:resp])[1]


m23 = @model begin
    @param   θ ∈ VectorDomain(12)
    @random  η ~ MvNormal(Matrix{Float64}(I, 11, 11))

    @pre begin
        Ka1     = θ[1]
        CL      = θ[2]*exp(η[1])
        Vc      = θ[3]*exp(η[2])
        Q       = θ[4]*exp(η[3])
        Vp      = θ[5]*exp(η[4])
        Kin     = θ[6]*exp(η[5])
        Kout    = θ[7]*exp(η[6])
        IC50    = θ[8]*exp(η[7])
        IMAX    = θ[9]*exp(η[8])
        γ       = θ[10]*exp(η[9])
        Vmax    = θ[11]*exp(η[10])
        Km      = θ[12]*exp(η[11])
    end

    @init begin
        Resp = θ[6]/θ[7]
    end

    @dynamics begin
        # TODO: allow intermediate expressions in macro
        Ev1'    = -Ka1*Ev1
        Cent'   =  Ka1*Ev1 - (CL+Vmax/(Km+(Cent/Vc))+Q)*(Cent/Vc)  + Q*(Periph/Vp)
        Periph' =  Q*(Cent/Vc)  - Q*(Periph/Vp)
        Resp'   =  Kin*(1-(IMAX*(Cent/Vc)^γ/(IC50^γ+(Cent/Vc)^γ)))  - Kout*Resp
    end

    @derived begin
        # TODO: allow direct output of dynamical variables
        ev1    = Ev1
        cp     = Cent / θ[3]
        periph = Periph
        resp   = Resp
    end
end


param = (θ = [
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
              ],)
randeffs = (η = zeros(11),)
sim = simobs(m23, subject, param, randeffs, abstol=1e-12,reltol=1e-12, continuity=:left)

# exclude discontinuities
inds = vcat(1:240,242:480,482:720,722:length(subject.observations))

@test sim[:ev1][inds] ≈ subject.observations.ev1[inds]
@test sim[:cp] ≈ subject.observations.cp rtol=1e-6
@test sim[:periph] ≈ subject.observations.periph rtol=1e-6
@test sim[:resp] ≈ subject.observations.resp rtol=1e-6

# Indirect Response Model (irm2)

###############################
# Test 24
###############################

# ev24 - Testing irm2
# - Indirect response model, type 2
# - Inhibition of response elimination
# - Two-compartment PK model
# - Optional nonlinear clearance - not being used in this example (Vmax and Km)
# use ev24.csv in PuMaS/examples/event_data/
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

subject = process_nmtran(example_nmtran_data("event_data/data24"),
                         [], [:ev1,:cp,:periph,:resp])[1]


m24 = @model begin
    @param   θ ∈ VectorDomain(12)
    @random  η ~ MvNormal(Matrix{Float64}(I, 11, 11))

    @pre begin
        Ka1     = θ[1]
        CL      = θ[2]*exp(η[1])
        Vc      = θ[3]*exp(η[2])
        Q       = θ[4]*exp(η[3])
        Vp      = θ[5]*exp(η[4])
        Kin     = θ[6]*exp(η[5])
        Kout    = θ[7]*exp(η[6])
        IC50    = θ[8]*exp(η[7])
        IMAX    = θ[9]*exp(η[8])
        γ       = θ[10]*exp(η[9])
        Vmax    = θ[11]*exp(η[10])
        Km      = θ[12]*exp(η[11])
    end

    @init begin
        Resp = θ[6]/θ[7]
    end

    @dynamics begin
        # TODO: allow intermediate expressions in macro
        Ev1'    = -Ka1*Ev1
        Cent'   =  Ka1*Ev1 - (CL+Vmax/(Km+(Cent/Vc))+Q)*(Cent/Vc)  + Q*(Periph/Vp)
        Periph' =  Q*(Cent/Vc)  - Q*(Periph/Vp)
        Resp'   =  Kin - Kout*(1-(IMAX*(Cent/Vc)^γ/(IC50^γ+(Cent/Vc)^γ)))*Resp
    end

    @derived begin
        # TODO: allow direct output of dynamical variables
        ev1    = Ev1
        cp     = Cent / θ[3]
        periph = Periph
        resp   = Resp

    end
end


sim = simobs(m24, subject, param, randeffs, abstol=1e-12,reltol=1e-12, continuity=:left)

# exclude discontinuities
inds = vcat(1:240,242:480,482:720,722:length(subject.observations))

@test sim[:ev1][inds] ≈ subject.observations.ev1[inds]
@test sim[:cp] ≈ subject.observations.cp rtol=1e-6
@test sim[:periph] ≈ subject.observations.periph  rtol=1e-6
@test sim[:resp] ≈ subject.observations.resp rtol=1e-6
