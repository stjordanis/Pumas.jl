## Types

"""
    Subject

The data corresponding to a single subject:

Fields:
- `id::Int`: numerical identifier
- `obsverations`: a vector of `Observation`s
- `covariates`: a named tuple containing the covariates, or `nothing`.
- `events`: a vector of `Event`s.
"""
struct Subject{T1,T2,T3}
  id::Int
  observations::T1
  covariates::T2
  events::T3
end

"""
    Population

A set of `Subject`s.
"""
struct Population{T} <: AbstractVector{T}
  subjects::T
end

"""
    Observation

A single measurement at a point in time.

Fields
- `time`: Time of measurement
- `val`: value of measurement; this will typically be a named tuple.
- `cmt`: Compartment from which measurement was taken
"""
struct Observation{T,V,C}
    time::T
    val::V
    cmt::C
end


"""
    Event

A single event

Fields:
 * `amt`: Dose amount (mass units, e.g. g, mg, etc.)
 * `time`: Time of dose (time unit, e.g. hours, days, etc.)
 * `evid`: Event identifier, possible values are
   - `-1`: End of an infusion dose
   - `0`: observation (these should have already been removed)
   - `1`: dose event
   - `2`: other type event
   - `3`: reset event (the amounts in each compartment are reset to zero, the on/off status of each compartment is reset to its initial status)
   - `4`: reset and dose event
 * `cmt`: Compartment
 * `rate`: The dosing rate:
   - `0`: instantaneous bolus dose
   - `>0`: infusion dose (mass/time units, e.g. mg/hr)
   - `-1`: infusion rate specifed by model
   - `-2`: infustion duration specified by model
 * `duration`: duration of dose, if `dose > 0`.
 * `ss`: steady state status
   - `0`: dose is not a steady state dose.
   - `1`: dose is a steady state dose, and that the compartment amounts are to be reset to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled.
   - `2`: dose is a steady state dose and that the compartment amounts are to be set to the sum of the steady-state amounts resulting from the given dose plus whatever those amounts would have been at the event time were the steady-state dose not given.
 * `ii`: interdose interval (units: time). Time between implied or additional doses.
 * `base_time`: time at which infusion started
 * `rate_dir`: direction of rate
   - `-1` for end-of-infusion
   - `+1` for any other doses
"""
struct Event{T,T2,T3} # Split parameters for dual numbers
  amt::T
  time::T2
  evid::Int8
  cmt::Int
  rate::T
  duration::T
  ss::Int8
  ii::T3
  base_time::T2 # So that this is kept after modifications to duration and rate
  rate_dir::Int8
end

Event(amt, time, evid, cmt) = Event(amt, time, evid, cmt, 0.0, 0.0, Int8(0), 0.0, time, Int8(1))



Base.isless(a::Event,b::Event) = isless(a.time,b.time)
Base.isless(a::Event,b::Number) = isless(a.time,b)
Base.isless(a::Number,b::Event) = isless(a,b.time)


function timespan(sub::Subject)
    lo, hi = extrema(evt.time for evt in sub.events)
    if !isempty(sub.observations)
        obs_lo, obs_hi = extrema(obs.time for obs in sub.observations)
        lo = min(lo, obs_lo)
        hi = max(hi, obs_hi)
    end
    lo, hi
end

observationtimes(sub::Subject) = [obs.time for obs in sub.observations]
