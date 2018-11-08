## Types

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
Base.summary(::Observation) = "Observation"
function Base.show(io::IO, o::Observation)
  println(io, summary(o))
  println(io, "  time of measurement = $(o.time)")
  if o.cmt == nothing
    println(io, "  no compartment specified")
  else
    println(io, "  compartment = $(o.cmt)")
  end
  println(io, "  measurements")
  foreach(v -> println(io, "    $(v) = $(getfield(o.val, v))"),
          fieldnames(typeof(o.val)))
end
TreeViews.hastreeview(::Observation) = true
function TreeViews.treelabel(io::IO, o::Observation, mime::MIME"text/plain")
  show(io, mime, Text(summary(o)))
end

abstract type AbstractEvent end
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
struct Event{T,T2,T3} <: AbstractEvent
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

Event(amt, time, evid, cmt) = Event(amt, time, evid, cmt, zero(amt), zero(amt),
                                    Int8(0), 0.0, time, Int8(1))

"""
  EventProxy

A proxy `Event` object with updated `time`, `rate` and `duration`.
Created by the solvers when non-default "magic arguments" (lags,bioav,rate,duration)
changes the properties of an event. In particular, the updated quantities can all be
dual numbers when doing derivative calculations.
"""
struct EventProxy{E<:Event,tType,rType,dType} <: AbstractEvent
  ev::E # reference to the original event
  time::tType
  rate::rType
  duration::dType
end
Base.propertynames(proxy::EventProxy) = propertynames(proxy.ev)
function Base.getproperty(proxy::EventProxy, name::Symbol)
  if name in fieldnames(EventProxy)
    getfield(proxy, name)
  else
    getproperty(proxy.ev, name)
  end
end

Base.isless(a::AbstractEvent,b::AbstractEvent) = isless(a.time,b.time)
Base.isless(a::AbstractEvent,b::Number) = isless(a.time,b)
Base.isless(a::Number,b::AbstractEvent) = isless(a,b.time)

### Display
_evid_index = Dict(-1 => "End of infusion", 0 => "Observation", 1 => "Dose",
                   2 => "Other", 3 => "Reset", 4 => "Reset and dose")
Base.summary(e::AbstractEvent) = "$(_evid_index[e.evid]) event"
function Base.show(io::IO, e::AbstractEvent)
  println(io, summary(e))
  println(io, "  dose amount = $(e.amt)")
  println(io, "  dose time = $(e.time)")
  println(io, "  compartment = $(e.cmt)")
  if e.rate > 0
    println(io, "  rate = $(e.rate)")
    println(io, "  duration = $(e.duration)")
  elseif e.rate == 0
    println(io, "  instantaneous")
  elseif e.rate == -1
    println(io, "  rate specified by model")
  elseif e.rate == -2
    println(io, "  duration specified by model")
  end
  if e.ss == 1
    println(io, "  steady state dose")
  elseif e.ss == 2
    println(io, "  steady state dose, no reset")
  end
  println(io, "  interdose interval = $(e.ii)")
  println(io, "  infusion start time = $(e.base_time)")
  if e.rate_dir == -1
    println(io, "  end of infusion")
  end
end
TreeViews.hastreeview(::AbstractEvent) = true
function TreeViews.treelabel(io::IO, e::AbstractEvent, mime::MIME"text/plain")
  show(io, mime, Text(summary(e)))
end

"""
    DosageRegimen

Lazy representation of a series of Events.
"""
mutable struct DosageRegimen
  data::DataFrame
  function DosageRegimen(amt::Number,
                         time::Number,
                         cmt::Number,
                         evid::Number,
                         ii::Number,
                         addl::Number,
                         rate::Number,
                         ss::Number)
    amt = isa(amt, Unitful.Mass) ?
    convert(Float64, getfield(uconvert.(u"mg", amt), :val)) :
    float(amt)
    time = isa(time, Unitful.Time) ?
    convert(Float64, getfield(uconvert.(u"hr", time), :val)) :
    float(time)
    cmt = convert(Int, cmt)
    evid = convert(Int8, evid)
    ii = isa(ii, Unitful.Time) ?
    convert(Float64, getfield(uconvert.(u"hr", ii), :val)) :
    float(ii)
    addl = convert(Int, addl)
    rate = convert(Float64, rate)
    ss = convert(Int8, ss)
    amt > 0 || throw(ArgumentError("amt must be non-negative"))
    time ≥ 0 || throw(ArgumentError("time must be non-negative"))
    evid == 0 && throw(ArgumentError("observations are not allowed"))
    evid ∈ 1:4 || throw(ArgumentError("evid must be a valid event type"))
    ii ≥ 0 || throw(ArgumentError("ii must be non-negative"))
    addl ≥ 0 || throw(ArgumentError("addl must be non-negative"))
    addl > 0 && ii == 0 && throw(ArgumentError("ii must be positive for addl > 0"))
    rate ∈ -2:0 || rate .> 0 || throw(ArgumentError("rate is invalid"))
    ss ∈ 0:2 || throw(ArgumentError("ss is invalid"))
    return new(DataFrame(time = time, cmt = cmt, amt = amt,
                         evid = evid, ii = ii, addl = addl,
                         rate = rate, ss = ss))
  end
  DosageRegimen(amt::Numeric;
                time::Numeric = 0,
                cmt::Numeric  = 1,
                evid::Numeric = 1,
                ii::Numeric   = 0,
                addl::Numeric = 0,
                rate::Numeric = 0,
                ss::Numeric   = 0) =
  DosageRegimen(DosageRegimen.(amt, time, cmt, evid, ii, addl,
                               rate, ss))
  DosageRegimen(regimen::DosageRegimen) = regimen
  function DosageRegimen(regimen1::DosageRegimen,
                         regimen2::DosageRegimen;
                         offset = nothing)
    data1 = getfield(regimen1, :data)
    data2 = getfield(regimen2, :data)
    if offset === nothing
      output = sort!(vcat(data1, data2), :time)
    else
      data2 = deepcopy(data2)
      data2[:time] = cumsum(prepend!(data1[:ii][end] * (data1[:addl][end] + 1) +
                                     data1[:time][end] +
                                     offset))
      output = sort!(vcat(data1, data2), :time)
    end
    return new(output)
  end
  DosageRegimen(regimens) =
  reduce((x, y) -> DosageRegimen(x, y), regimens)
end

"""
    Subject

The data corresponding to a single subject:

Fields:
- `id::Int`: numerical identifier
- `observations`: a vector of `Observation`s
- `covariates`: a named tuple containing the covariates, or `nothing`.
- `events`: a vector of `Event`s.
"""
struct Subject{T1,T2,T3}
  id::Int
  observations::T1
  covariates::T2
  events::T3
  function Subject(;id = 1,
                   obs = Observation[],
                   cvs = nothing,
                   evs = Event[])
    obs = build_observation_list(obs)
    evs = build_event_list(evs)
    return new{typeof(obs),typeof(cvs),typeof(evs)}(id, obs, cvs, evs)
  end
end

### Display
Base.summary(::Subject) = "Subject"
function Base.show(io::IO, subject::Subject)
  println(io, summary(subject))
  println(io, "  Events: ", length(subject.events))
  println(io, "  Observations: ", length(subject.observations))
  println(io, "  Covariates: ")
  subject.covariates !== nothing &&
  foreach(kv -> println(io, string("    ", kv[1], ": ", kv[2])),
          pairs(subject.covariates))
  !isempty(subject.observations) &&
  println(io, "  Observables: ",
          join(fieldnames(typeof(subject.observations[1].val)),", "))
  return nothing
end
TreeViews.hastreeview(::Subject) = true
function TreeViews.treelabel(io::IO, subject::Subject, mime::MIME"text/plain")
  show(io, mime, Text(summary(subject)))
end

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

"""
    Population(::AbstractVector{<:Subject})

A `Population` is a set of `Subject`s. It can be instanced passing a collection or `Subject`.
"""
struct Population{T} <: AbstractVector{T}
  subjects::Vector{T}
  Population(obj::AbstractVector{T}) where {T<:Subject} = new{T}(obj)
end

### Display
Base.summary(::Population) = "Population"
function Base.show(io::IO, data::Population)
  println(io, summary(data))
  println(io, "  Subjects: ", length(data.subjects))
  if isassigned(data.subjects, 1)
    co = data.subjects[1].covariates
    co != nothing && println(io, "  Covariates: ", join(fieldnames(typeof(co)),", "))
    obs = data.subjects[1].observations
    !isempty(obs) && println(io, "  Observables: ", join(fieldnames(typeof(obs[1].val)),", "))
  end
  return nothing
end
TreeViews.hastreeview(::Population) = true
function TreeViews.treelabel(io::IO, data::Population, mime::MIME"text/plain")
  show(io, mime, Text(summary(data)))
end
