"""
  to_nt(obj)::NamedTuple{PN,VT}

It returns a NamedTuple based on the propertynames of the object.
If a value is a vector with a single value, it returns the value.
If the vector has no missing values, it is promoted through disallowmissing.
"""
to_nt(obj::Any) = propertynames(obj) |>
  (x -> NamedTuple{Tuple(x)}(
    getproperty(obj, x) |>
    (x -> isone(length(unique(x))) ?
          first(x) :
          any(ismissing, x) ? x : disallowmissing(x))
    for x ∈ x))

"""
    process_nmtran(filepath::String, args...; kwargs...)
    process_nmtran(data, cvs=Symbol[], dvs=Symbol[:dv];
                   id=:id, time=:time, evid=:evid, amt=:amt, addl=:addl,
                   ii=:ii, cmt=:cmt, rate=:rate, ss=:ss,
                   event_data = true)

Import NMTRAN-formatted data.

- `cvs` covariates specified by either names or column numbers
- `dvs` dependent variables specified by either names or column numbers
- `event_data` toggles assertions applicable to event data
"""
function process_nmtran(filepath::AbstractString, args...; kwargs...)
  process_nmtran(CSV.read(filepath, missingstrings=["."]), args...; kwargs...)
end
function process_nmtran(data,cvs=Symbol[],dvs=Symbol[:dv];
                        id=:id, time=:time, evid=:evid, amt=:amt, addl=:addl,
                        ii=:ii, cmt=:cmt, rate=:rate, ss=:ss,
                        mdv=:mdv, event_data = true)
  data = copy(data)
  colnames = names(data)

  if id ∉ colnames
    data[id] = "1"
  end
  if time ∉ colnames
    data[time] = 0.0
  end
  if evid ∉ colnames
    data[evid] = Int8(0)
  end
  if mdv ∉ colnames
    data[mdv] = Int8(0)
  end
  if cvs isa AbstractVector{<:Integer}
    cvs = colnames[cvs]
  end
  if dvs isa AbstractVector{<:Integer}
    dvs = colnames[dvs]
  end
  allowmissing!(data, dvs)
  mdv = isone.(data[mdv])
  for dv in dvs
    data[dv] .= ifelse.(mdv, missing, data[dv])
  end
  Subject.(groupby(data, id), Ref(colnames), id, time, evid, amt, addl, ii, cmt,
           rate, ss, Ref(cvs), Ref(dvs), event_data)
end

function build_observation_list(obs::AbstractDataFrame)
  #cmt = :cmt ∈ names(obs) ? obs[:cmt] : 1
  vars = setdiff(names(obs), (:time, :cmt))
  return NamedTuple{ntuple(i->vars[i],length(vars))}(ntuple(i -> convert(AbstractVector{Union{Missing,Float64}}, obs[vars[i]]), length(vars)))
end
build_observation_list(obs::NamedTuple) = obs
build_observation_list(obs::Nothing) = obs

build_event_list(evs::AbstractVector{<:Event}, event_data::Bool) = evs
function build_event_list!(events, event_data, t, evid, amt, addl, ii, cmt, rate, ss)
  @assert evid ∈ 0:4 "evid must be in 0:4"
  # Dose-related data items
  drdi = iszero(amt) && (rate == 0) && iszero(ii) && iszero(addl) && iszero(ss)
  if event_data
    if evid ∈ [0, 2, 3]
      @assert drdi "Dose-related data items must be zero when evid = $evid"
    else
      @assert !drdi "Some dose-related data items must be non-zero when evid = $evid"
    end
  end
  duration = amt / rate
  for j = 0:addl  # addl==0 means just once
    _ss = iszero(j) ? ss : zero(Int8)
    if iszero(amt) && evid ≠ 2
      # These are dose events having AMT=0, RATE>0, SS=1, and II=0.
      # Such an event consists of infusion with the stated rate,
      # starting at time −∞, and ending at the time on the dose
      # ev event record. Bioavailability fractions do not apply
      # to these doses.
      push!(events, Event(amt, t, evid, cmt, rate, ii, _ss, ii, t, Int8(1)))
    else
      push!(events, Event(amt, t, evid, cmt, rate, duration, _ss, ii, t, Int8(1)))
      if !iszero(rate) && iszero(_ss)
        push!(events, Event(amt, t + duration, Int8(-1), cmt, rate, duration, _ss, ii, t, Int8(-1)))
      end
    end
    t += ii
  end
end
function build_event_list(regimen::DosageRegimen, event_data::Bool)
  data = regimen.data
  events = Event[]
  for i in 1:size(data, 1)
    t    = data[:time][i]
    evid = data[:evid][i]
    amt  = data[:amt][i]
    addl = data[:addl][i]
    ii   = data[:ii][i]
    cmt  = data[:cmt][i]
    rate = data[:rate][i]
    ss   = data[:ss][i]
    build_event_list!(events, event_data, t, evid, amt, addl, ii, cmt, rate, ss)
  end
  sort!(events)
end
