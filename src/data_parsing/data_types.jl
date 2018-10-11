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

Event(amt, time, evid, cmt) = Event(amt, time, evid, cmt, zero(amt), zero(amt),
                                    Int8(0), 0.0, time, Int8(1))

Base.isless(a::Event,b::Event) = isless(a.time,b.time)
Base.isless(a::Event,b::Number) = isless(a.time,b)
Base.isless(a::Number,b::Event) = isless(a,b.time)

function Base.show(io::IO, e::Event)
    evid = Dict(-1 => "End of infusion", 0 => "Observation", 1 => "Dose", 
                2 => "Other", 3 => "Reset", 4 => "Reset and dose")
    rate = Dict(0 => "instantaneous", -1 => "rate specified by model",
                -2 => "duration specified by model")
    rate_dir = Dict(-1 => "end of infusion", 1 => "other")
    println(io, "$(evid[e.evid]) event: ")
    println(io, "  dose amount = $(e.amt)")
    println(io, "  dose time = $(e.time)")
    println(io, "  compartment = $(e.cmt)")
    if e.rate > 0
        println(io, "  rate = $(e.rate)")
        println(io, "  duration = $(e.duration)")
    else
        println(io, "  $(rate[e.rate])")
    end
    if e.ss != 0
        println(io, "  steady state, type = $(e.ss)")
    end
    println(io, "  interdose interval = $(e.ii)")
    println(io, "  base time = $(e.base_time)")
    println(io, "  rate direction = $(rate_dir[e.rate_dir])")
end

"""
    DosageRegimen

Lazy representation of a series of Events.
"""
mutable struct DosageRegimen
    data::DataFrame
    function DosageRegimen(amt::Union{<:Unitful.Mass,<:Real};
                           time::Union{<:Unitful.Time,Real} = 0.,
                           cmt::Number      = 1,
                           evid::Number     = 1,
                           ii::Union{<:Unitful.Time,Real}   = 0.,
                           addl::Number     = 0,
                           rate::Number     = 0.,
                           ss::Number       = 0,
                           n::Number        = 1)
        amt = isa(amt, Unitful.Mass) ?
            convert(Float64, getfield(uconvert.(u"g", amt), :val)) :
            float(amt)
        time = isa(time, Unitful.Time) ?
            convert(Float64, getfield(uconvert.(u"hr", time), :val)) :
            float(time)
        cmt = convert(Int, cmt)
        evid = convert(Int, evid)
        ii = isa(ii, Unitful.Time) ?
            convert(Float64, getfield(uconvert.(u"hr", ii), :val)) :
            float(ii)
        addl = convert(Int, addl)
        rate = convert(Float64, rate)
        ss = convert(Int, ss)
        n = convert(Int, n)
        amt > 0 || throw(ArgumentError("amt must be non-negative"))
        time ≥ 0 || throw(ArgumentError("time must be non-negative"))
        evid == 0 && throw(ArgumentError("observations are not allowed"))
        evid ∈ 1:4 || throw(ArgumentError("evid must be a valid event type"))
        ii ≥ 0 || throw(ArgumentError("ii must be non-negative"))
        addl ≥ 0 || throw(ArgumentError("addl must be non-negative"))
        addl > 0 && ii == 0 && throw(ArgumentError("ii must be positive for addl > 0"))
        rate ∈ -2:0 || rate .> 0 || throw(ArgumentError("rate is invalid"))
        ss ∈ 0:2 || throw(ArgumentError("ss is invalid"))
        n > 0 || throw(ArgumentError("n must be positive"))
        return new(map(col -> repeat(col, n),
                       eachcol(DataFrame(time = time, cmt = cmt, amt = amt,
                                         evid = evid, ii = ii, addl = addl,
                                         rate = rate, ss = ss))))
    end
    function DosageRegimen(regimen1::DosageRegimen,
                           regimen2::DosageRegimen;
                           wait::Union{Nothing,Union{<:Unitful.Time,Real}} = nothing)
        data1 = getfield(regimen1, :data)
        data2 = getfield(regimen2, :data)
        if wait === nothing
            output = sort!(vcat(data1, data2), :time)
        else
            data2 = deepcopy(data2)
            data2[:time] = cumsum(prepend!(diff(data2[:time]),
                                           data1[:ii][end] * (data1[:addl][end] + 1) +
                                           data1[:time][end] +
                                           wait))
            output = sort!(vcat(data1, data2), :time)
        end
        return new(output)
    end
    function DosageRegimen(regimen::DosageRegimen;
                           wait::Union{Nothing,Union{<:Unitful.Time,Real}} = nothing,
                           n::Integer = 1)
        output = regimen
        for each in 2:n
            output = DosageRegimen(output, regimen, wait = wait)
        end
        return output
    end
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
end

function Base.show(io::IO, subject::Subject)
    println(io, "Subject")
    println(io, "  Events: ", length(subject.events))
    println(io, "  Observations: ", length(subject.observations))
    println(io, "  Covariates: ", join(fieldnames(typeof(subject.covariates)),", "))
    !isempty(subject.observations) &&
        println(io, "  Observables: ",
                    join(fieldnames(typeof(subject.observations[1].val)),", "))
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
    Population(;
               cvs::Dict{<:Integer,<:NamedTuple} = Dict{Int,NamedTuple}(),
               obs::AbstractDataFrame = DataFrame(),
               evs::Dict{<:Integer,<:DosageRegimen} = Dict{Int,DosageRegimen}()))

A `Population` is a set of `Subject`s. It can be instanced passing covariates,
observations, and event data for each relevant `Subject` through a unique id.

# Examples
```jldoctest
julia> population = c = Population(cvs = Dict(1:4 .=>
                                              repeat(NamedTuple{(:sex,:age)}.(zip([:male,:female],
                                                                                  [25, 27])),
                                                     outer = 2)),
                                   evs = Dict(1:4 .=>
                                              (DosageRegimen(amt, time = time) for
                                                   amt ∈ 1u"g":1u"g":2u"g" for
                                                   time ∈ 30u"minute":1u"hr":1.5u"hr")),
                                   obs = DataFrame(id = 1:4,
                                                   time = 0u"hr":1u"hr":3u"hr",
                                                   variable = "dvs",
                                                   value = 1u"g":2u"g":7u"g",
                                                   ),
                                   )
Population
  Subjects: 4
  Covariates: sex, age
  Observables: dvs
```
"""
struct Population{T} <: AbstractVector{T}
    subjects::Vector{T}
    Population(obj::AbstractVector{T}) where {T<:Subject} = new{T}(obj)
    function Population(;
                        cvs::Dict{<:Integer,<:NamedTuple} = Dict{Int,NamedTuple}(),
                        obs::AbstractDataFrame = DataFrame(),
                        evs::Dict{<:Integer,<:DosageRegimen} = Dict{Int,DosageRegimen}())
        ids = sort!(collect(union(keys(cvs), keys(evs),
            isempty(obs) ? Set{Int}() : obs[:id])))
        if :cmt ∉ names(obs) && !isempty(obs)
            obs[:cmt] = nothing
        end
        if !isempty(obs)
            Set(names(obs)) == Set((:id, :cmt, :time, :variable, :value)) ||
                throw(ArgumentError("Observations should have id, time, variable, and value"))
            if isa(obs[:time], AbstractVector{<:Unitful.Time})
                obs[:time] = convert.(Float64,
                                      getfield.(uconvert.(u"hr", obs[:time]),
                                                :val))
            end
            if isa(obs[:value], AbstractVector{<:Unitful.Mass})
                obs[:value] = convert.(Float64,
                                       getfield.(uconvert.(u"g", obs[:value]),
                                                 :val))
            end
        end
        regimens =
            Dict((regimen => process_data(regimen)) for
                  regimen ∈ unique(values(evs)))
        output = Vector{Subject}(undef, length(ids))
        for (idx, id) ∈ enumerate(ids)
            output[idx] = Subject(id,
                                  process_data(obs, id),
                                  get(cvs, id, nothing),
                                  id ∈ keys(evs) ? regimens[evs[id]] : Vector{Event}())
        end
        return new{eltype(output)}(output)
    end
end

function Base.show(io::IO, data::Population)
    typeof(output)
    println(io, "Population")
    println(io, "  Subjects: ", length(data.subjects))
    if isassigned(data.subjects, 1)
        co = data.subjects[1].covariates
        co != nothing && println(io, "  Covariates: ", join(fieldnames(typeof(co)),", "))
        obs = data.subjects[1].observations
        !isempty(obs) && println(io, "  Observables: ", join(fieldnames(typeof(obs[1].val)),", "))
    end
    nothing
end
