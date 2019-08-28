## Types

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
 * `cmt`: Compartment (an Int or Symbol)
 * `rate`: The dosing rate:
   - `0`: instantaneous bolus dose
   - `>0`: infusion dose (mass/time units, e.g. mg/hr)
   - `-1`: infusion rate specifed by model
   - `-2`: infusion duration specified by model
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
struct Event{T1,T2,T3,T4,T5,T6,C}
  amt::T1
  time::T2
  evid::Int8
  cmt::C
  rate::T3
  duration::T4
  ss::Int8
  ii::T5
  base_time::T6 # So that this is kept after modifications to duration and rate
  rate_dir::Int8
end

Event(amt, time, evid::Integer, cmt::Union{Int,Symbol}) =
  Event(amt, time, Int8(evid), cmt, zero(amt), zero(amt),
        Int8(0), zero(time), time, Int8(1))

Base.isless(a::Event,b::Event) = isless(a.time,b.time)
Base.isless(a::Event,b::Number) = isless(a.time,b)
Base.isless(a::Number,b::Event) = isless(a,b.time)

### Display
_evid_index = Dict(-1 => "End of infusion", 0 => "Observation", 1 => "Dose",
                   2 => "Other", 3 => "Reset", 4 => "Reset and dose")
Base.summary(e::Event) = "$(_evid_index[e.evid]) event"
function Base.show(io::IO, e::Event)
  println(io, summary(e))
  println(io, "  dose amount = $(e.amt)")
  println(io, "  dose time = $(e.time)")
  println(io, "  compartment = $(e.cmt)")
  if e.rate > zero(e.rate)
    println(io, "  rate = $(e.rate)")
    println(io, "  duration = $(e.duration)")
  elseif e.rate == zero(e.rate)
    println(io, "  instantaneous")
  elseif e.rate == -oneunit(e.rate)
    println(io, "  rate specified by model")
  elseif e.rate == -2oneunit(e.rate)
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
TreeViews.hastreeview(::Event) = true
function TreeViews.treelabel(io::IO, e::Event, mime::MIME"text/plain")
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
                         cmt::Union{Number,Symbol},
                         evid::Number,
                         ii::Number,
                         addl::Number,
                         rate::Number,
                         duration::Number,
                         ss::Number)
    # amt = evid ∈ [0, 3] ? float(zero(amt)) : float(amt)
    amt = float(amt)
    time = float(time)
    cmt = cmt isa Symbol ? cmt : convert(Int, cmt)
    evid = convert(Int8, evid)
    ii = float(ii)
    addl = convert(Int, addl)
    rate = float(rate)
    ss = convert(Int8, ss)
    evid ∈ 1:4 || throw(ArgumentError("evid must be a valid event type"))
    if evid ∈ [0, 3]
      iszero(amt) || throw(ArgumentError("amt must be 0 for evid = $evid"))
    elseif iszero(amt) && rate > 0 && duration > 0
      amt = rate * duration
    else
      amt > zero(amt) || throw(ArgumentError("amt must be positive for evid = $evid"))
    end
    time ≥ zero(time) || throw(ArgumentError("time must be non-negative"))
    evid == 0 && throw(ArgumentError("observations are not allowed"))
    ii ≥ zero(ii) || throw(ArgumentError("ii must be non-negative"))
    addl ≥ 0 || throw(ArgumentError("addl must be non-negative"))
    addl > 0 && ii == zero(ii) && throw(ArgumentError("ii must be positive for addl > 0"))
    rate ≥ zero(rate) || rate == -2 || throw(ArgumentError("rate is invalid"))
    ss ∈ 0:2 || throw(ArgumentError("ss is invalid"))
    if iszero(duration) && amt > zero(amt) && rate > zero(rate)
      duration = amt / rate
    elseif iszero(rate) && amt > zero(amt) && duration > zero(rate)
      rate = amt / duration
    elseif duration > zero(duration) && rate > zero(rate)
      @assert amt ≈ rate * duration
    end
    new(DataFrame(time = time, cmt = cmt, amt = amt, evid = evid, ii = ii, addl = addl,
                  rate = rate, duration = duration, ss = ss))
  end
  DosageRegimen(amt::Numeric;
                time::Numeric = 0,
                cmt::Union{Numeric,Symbol} = 1,
                evid::Numeric = 1,
                ii::Numeric = zero.(time),
                addl::Numeric = 0,
                rate::Numeric = zero.(amt)./oneunit.(time),
                duration::Numeric = zero(amt)./oneunit.(time),
                ss::Numeric = 0) =
  DosageRegimen(DosageRegimen.(amt, time, cmt, evid, ii, addl, rate, duration, ss))
  DosageRegimen(regimen::DosageRegimen) = regimen
  function DosageRegimen(regimen1::DosageRegimen,
                         regimen2::DosageRegimen;
                         offset = nothing)
    data1 = regimen1.data
    data2 = regimen2.data
    if isnothing(offset)
      output = sort!(vcat(data1, data2), :time)
    else
      data2 = deepcopy(data2)
      data2[!,:time] = cumsum(prepend!(data1[!,:ii][end] * (data1[!,:addl][end] + 1) +
                                       data1[!,:time][end] +
                                       offset))
      output = sort!(vcat(data1, data2), :time)
    end
    new(output)
  end
  DosageRegimen(regimens::AbstractVector{<:DosageRegimen}) = reduce(DosageRegimen, regimens)
  DosageRegimen(regimens::DosageRegimen...) = reduce(DosageRegimen, regimens)
end
"""
  DataFrame(evs::DosageRegimen, expand::Bool = false)

  Create a DataFrame with the information in the dosage regimen.
  If expand, creates a DataFrame with the information in the event list (expanded form).
"""
function DataFrames.DataFrame(evs::DosageRegimen, expand::Bool = false)
  !expand && return evs.data
  evs = build_event_list(evs, true)
  output = DataFrame(fill(Float64, 10),
                     [:amt, :time, :evid, :cmt, :rate, :duration, :ss, :ii, :base_time, :rate_dir],
                     length(evs))
  for col ∈ [:amt, :time, :evid, :cmt, :rate, :duration, :ss, :ii, :base_time, :rate_dir]
    output[col] .= getproperty.(evs, col)
  end
  sort!(output, [:time])
end

"""
    Subject

The data corresponding to a single subject:

Fields:
- `id`: identifier
- `observations`: a NamedTuple of the dependent variables
- `covariates`: a named tuple containing the covariates, or `nothing`.
- `events`: a vector of `Event`s.
- `time`: a vector of time stamps for the observations
"""
struct Subject{T1,T2,T3,T4}
  id::String
  observations::T1
  covariates::T2
  events::T3
  time::T4

  function Subject(data, Names,
                   id, time, evid, amt, addl, ii, cmt, rate, ss,
                   cvs = Symbol[], dvs = Symbol[:dv],
                   event_data = true)
    ## Observations
    idx_obs = findall(iszero, data[!,evid])
    obs_times = Missings.disallowmissing(data[!,time][idx_obs])
    @assert issorted(obs_times) "Time is not monotonically increasing within subject"
    if isa(obs_times, Unitful.Time)
      _obs_times = convert.(Float64, getfield(uconvert.(u"hr", obs_times), :val))
    else
      _obs_times = float(obs_times)
    end

    dv_idx_tuple = ntuple(i -> convert(AbstractVector{Union{Missing,Float64}},
                                       data[!,dvs[i]][idx_obs]),
                                       length(dvs))
    observations = NamedTuple{tuple(dvs...),typeof(dv_idx_tuple)}(dv_idx_tuple)

    # cmt handling should be reversed: it should give it the appropriate name given cmt
    # obs_cmts = :cmt ∈ Names ? data[:cmt][idx_obs] : nothing

    ## Covariates
    covariates = isempty(cvs) ? nothing : to_nt(data[!,cvs])


    ## Events
    idx_evt = setdiff(1:size(data, 1), idx_obs)

    n_amt = amt ∈ Names
    n_addl = addl ∈ Names
    n_ii = ii ∈ Names
    n_cmt = cmt ∈ Names
    n_rate = rate ∈ Names
    n_ss = ss ∈ Names
    cmtType = n_cmt ? (eltype(data[!,cmt]) <: String ? Symbol : Int) : Int
    events = Event{Float64,Float64,Float64,Float64,Float64,Float64,cmtType}[]
    for i in idx_evt
      t     = float(data[!,time][i])
      _evid = Int8(data[!,evid][i])
      _amt  = n_amt ? float(data[!,amt][i])   : 0. # can be missing if evid=2
      _addl = n_addl ? Int(data[!,addl][i])   : 0
      _ii   = n_ii ? float(data[!,ii][i])     : zero(t)
      __cmt  = n_cmt ? data[!,cmt][i] : 1
      _cmt = __cmt isa String ? Symbol(__cmt) : Int(__cmt)
      _rate = n_rate ? float(data[!,rate][i]) : _amt === nothing ? 0.0 : zero(_amt)/oneunit(t)
      ss′   = n_ss ? Int8(data[!,ss][i])      : Int8(0)
      build_event_list!(events, event_data, t, _evid, _amt, _addl, _ii, _cmt, _rate, ss′)
    end
    sort!(events)
    new{typeof(observations),typeof(covariates),typeof(events),typeof(_obs_times)}(
      string(first(data[!,id])), observations, covariates, events, _obs_times)
  end

  function Subject(;id = "1",
                   obs = nothing,
                   cvs = nothing,
                   evs = Event[],
                   time = obs isa AbstractDataFrame ? obs.time : nothing,
                   event_data = true,)
    obs = build_observation_list(obs)
    evs = build_event_list(evs, event_data)
    _time = isnothing(time) ? nothing : Missings.disallowmissing(time)
    @assert isnothing(time) || issorted(_time) "Time is not monotonically increasing within subject"
    new{typeof(obs),typeof(cvs),typeof(evs),typeof(_time)}(string(id), obs, cvs, evs, _time)
  end
end


function DataFrames.DataFrame(subject::Subject; include_covariates=true, include_dvs=true)
  # Build a DataFrame that holds the events
  df_events = DataFrame(build_event_list(subject.events, true))
  # Remove events with evid==-1
  df_events = df_events[df_events[!, :evid].!=-1,:]

  # We delete rate/duration if no infusions. Else, we delete duration or rate
  # as appropriate. TODO This will actually fail if used in the context of a
  # Population where some Subjects have duration specified and others have
  # rate specified.
  if all(x->iszero(x), df_events[Not(ismissing.(df_events[!, :rate])),:rate])
    # There are no infusions
    select!(df_events, Not(:rate))
    select!(df_events, Not(:duration))
  elseif all(x->x>zero(x), df_events[Not(ismissing.(df_events[!, :rate])),:rate])
    # There are infusions, and they're specified by rate
    select!(df_events, Not(:duration))
  else
    # There are infusions, and they're specified by duration
    select!(df_events, Not(:rate))
  end

  # Generate the name for the dependent variable in a manner consistent with
  # multiple dvs etc
  if !isnothing(subject.time)
    df = DataFrame(id = fill(subject.id, length(subject.time)), time=subject.time)
    df[!, :evid] .= 0
    # Only include the dv columns if include_dvs is specified and there are
    # observations.
    if include_dvs && !isnothing(subject.observations)
      # Loop over all dv's
      for (dv_name, dv_vals) in pairs(subject.observations)
        df[!, dv_name] .= dv_vals
      end
    end
    # Now, create columns in df_events with missings for the column names in
    # df but not in df_events
    if include_dvs
      for df_name in names(df)
        if df_name == :id
          df_events[!, df_name] .= subject.id
        elseif !(df_name == :time)
          if df_name ∉ names(df_events)
            df_events[!, df_name] .= missing
          end
        end
      end
      # ... and do the same for df
      for df_name in names(df_events)
        if df_name ∉ names(df)
          df[!, df_name] .= missing
        end
      end
    end
  end

  # If there are no observations, just go with df_events
  if isnothing(subject.time)
    df = df_events
  elseif include_dvs
    df = vcat(df, df_events)
  else
    df
  end
  if include_covariates
    if !isa(subject.covariates, Nothing)
      for (covariate, value) in pairs(subject.covariates)
        df[!,covariate] .= value
      end
    end
  end

  # Sort the df according to time first, and use :base_time to ensure that events
  # come before observations (they are missing for observations, so they will come
  # last).
  sort_bys = include_dvs ? (:time, :base_time) : (:time,)
  sort!(df, sort_bys)

  # Find the amount (amt) column, and insert dose and tad columns after it
  if include_dvs
    amt_pos = findfirst(isequal(:amt), names(df))
    insertcols!(df, amt_pos+1, :dose => 0.0)
    insertcols!(df, amt_pos+2, :tad => 0.0)
    # Calculate the indeces for the dose events
    idxs = findall(isequal(false), ismissing.(df[!, :amt]))
    # Append the number of rows to allow the +1 indexing used below
    if !(idxs[end] == nrow(df))
      push!(idxs, nrow(df))
    end
    for i in 1:length(idxs)-1
      df[idxs[i]:idxs[i+1], :dose] .= df[idxs[i], :amt]
      df[idxs[i]:idxs[i+1], :cmt] .= df[idxs[i], :cmt]
      df[idxs[i]:idxs[i+1], :tad] .= df[idxs[i]:idxs[i+1], :time].-df[idxs[i], :time]
    end
  end

  # Return df
  df
end


### Display
Base.summary(::Subject) = "Subject"
function Base.show(io::IO, subject::Subject)
  println(io, summary(subject))
  println(io, string("  ID: $(subject.id)"))
  evs = subject.events
  isnothing(evs) || println(io, "  Events: ", length(subject.events))
  obs = subject.observations
  observables = propertynames(obs)
  if !isempty(observables)
    vals = mapreduce(pn -> string(pn, ": (n=$(length(getindex(obs, pn))))"),
                     (x, y) -> "$x, $y",
                     observables)
    println(io, "  Observables: $vals")
  end
end
TreeViews.hastreeview(::Subject) = true
function TreeViews.treelabel(io::IO, subject::Subject, mime::MIME"text/plain")
  show(io, mime, Text(summary(subject)))
end

@recipe function f(obs::Subject; obsnames=nothing)
  t = obs.time
  names = Symbol[]
  plot_vars = []
  for (n,v) in pairs(obs.observations)
    if obsnames !== nothing
      !(n in obsnames) && continue
    end
    push!(names,n)
    push!(plot_vars,v)
  end
  xlabel --> "time"
  legend --> false
  lw --> 3
  ylabel --> reshape(names,1,length(names))
  layout --> good_layout(length(names))
  title --> "Subject ID: $(obs.id)"
  t,plot_vars
end

# Define Population as an alias
"""
A `Population` is an `AbstractVector` of `Subject`s.
"""
Population{T} = AbstractVector{T} where T<:Subject
Population(obj::Population...) = reduce(vcat, obj)::Population

function DataFrames.DataFrame(pop::Population; include_covariates=true, include_dvs=true)
  vcat((DataFrame(subject; include_covariates=include_covariates, include_dvs=include_dvs) for subject in pop)...)
end

### Display
Base.summary(::Population) = "Population"
function Base.show(io::IO, ::MIME"text/plain", population::Population)
  println(io, summary(population))
  println(io, "  Subjects: ", length(population))
  if isassigned(population, 1)
    co = population[1].covariates
    !isnothing(co) && println(io, "  Covariates: ", join(fieldnames(typeof(co)),", "))
    obs = population[1].observations
    !isnothing(obs) && println(io, "  Observables: ", join(keys(obs),", "))
  end
  return nothing
end
TreeViews.hastreeview(::Population) = true
TreeViews.numberofnodes(population::Population) = length(population)
TreeViews.treenode(population::Population, i::Integer) = getindex(population, i)
function TreeViews.treelabel(io::IO, population::Population, mime::MIME"text/plain")
  show(io, mime, Text(summary(population)))
end
TreeViews.nodelabel(io::IO, population::Population, i::Integer, mime::MIME"text/plain") = show(io, mime, Text(population[i].id))

@recipe function f(pop::Population)
  for p in pop
    @series begin
      lw --> 1.5
      title --> "Population Simulation"
      p
    end
  end
  nothing
end

# Convert to NCA types
import .NCA: NCAPopulation, NCASubject, NCADose
using .NCA: Formulation, IVBolus, IVInfusion, EV
function Base.convert(::Type{NCADose}, ev::Event)
  (ev.evid == Int8(0) || ev.evid == Int8(2) || ev.evid == Int8(3)) && return nothing
  time = ev.time
  amt = ev.amt
  duration = isinf(ev.duration) ? zero(ev.duration) : ev.duration
  NCADose(time, amt, duration, IVBolus) # FIXME: when is an event extravascular?
end
NCADose(dose::Event) = convert(NCADose, dose)

function Base.convert(::Type{NCASubject}, subj::Subject; name=:dv, kwargs...)
  dose = convert.(NCADose, subj.events)
  ii = subj.events[end].ii
  (subj.observations === nothing || subj.time === nothing) && return NCASubject(Float64[], Float64[]; id=subj.id, dose=nothing, clean=false, ii=ii, kwargs...)
  NCASubject(map(obs->obs.name, subj.observations), subj.time; clean=false, id=subj.id, dose=dose, ii=ii, kwargs...)
end
NCASubject(subj::Subject; name=:dv) = convert(NCASubject, subj; name=name)

Base.convert(::Type{NCAPopulation}, population::Population; name=:dv, kwargs...) =
  NCAPopulation(map(subject -> convert(NCASubject, subject; name=name, kwargs...), population))
NCAPopulation(population::Population; name=:dv, kwargs...) = convert(NCAPopulation, population; name=name, kwargs...)
