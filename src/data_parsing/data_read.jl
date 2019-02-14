Base.iterate(A::Population)    = iterate(A.subjects)
Base.iterate(A::Population, i) = iterate(A.subjects, i)

# size
Base.length(A::Population) = length(A.subjects)
Base.size(A::Population) = size(A.subjects)

# indexing
@inline function Base.getindex(A::Population, I...)
  @boundscheck checkbounds(A.subjects, I...)
  @inbounds return A.subjects[I...]
end
@inline function Base.setindex!(A::Population, x, I...)
  @boundscheck checkbounds(A.subjects, I...)
  @inbounds A.subjects[I...] = x
end
Base.axes(A::Population) = axes(A.subjects)
Base.IndexStyle(::Type{<:Population}) = Base.IndexLinear()

function process_nmtran(filepath::String,kwargs...)
  process_nmtran(CSV.read(filepath),kwargs...)
end

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
    process_nmtran(filename, cvs=[], dvs=[:dv])
    process_nmtran(data, cvs=[], dvs=[:dv])

Import NMTRAN-formatted data.

- `cvs` covariates specified by either names or column numbers
- `dvs` dependent variables specified by either names or column numbers

"""
function process_nmtran(data,cvs=Symbol[],dvs=Symbol[:dv];
                        id=:id, time=:time, evid=:evid, amt=:amt, addl=:addl,
                        ii=:ii, cmt=:cmt, rate=:rate, ss=:ss)
  Names = names(data)

  if id ∉ Names
    data[id] = 1
  end
  if time ∉ Names
    data[time] = 0.0
  end
  if evid ∉ Names
    data[evid] = Int8(1)
  end
  if cvs isa AbstractVector{<:Integer}
    cvs = Names[cvs]
  end
  if dvs isa AbstractVector{<:Integer}
    dvs = Names[dvs]
  end
  Population(Subject.(groupby(data, :id), Ref(Names),
                      id, time, evid, amt, addl, ii, cmt, rate, ss,
                      Ref(cvs), Ref(dvs)))
end
build_observation_list(obs::AbstractVector{<:Observation}) = obs
function build_observation_list(obs::AbstractDataFrame)
  isempty(obs) && return Observation[]
  #cmt = :cmt ∈ names(obs) ? obs[:cmt] : 1
  time = obs[:time]
  vars = setdiff(names(obs), (:time, :cmt))
  if isa(time, Unitful.Time)
    time = convert.(Float64, getfield(uconvert.(u"hr", time), :val))
  else
    time = float(time)
  end
  return Observation.(time,
                      NamedTuple{Tuple(vars),NTuple{length(vars),Float64}}.(values.(eachrow(obs[vars]))))
end

build_event_list(evs::AbstractVector{<:Event}) = evs
function build_event_list(regimen::DosageRegimen)
  data = getfield(regimen, :data)
  events = []
  for i in 1:size(data, 1)
    t    = data[:time][i]
    evid = data[:evid][i]
    amt  = data[:amt][i]
    addl = data[:addl][i]
    ii   = data[:ii][i]
    cmt  = data[:cmt][i]
    rate = data[:rate][i]
    ss   = data[:ss][i]

    for j = 0:addl  # addl==0 means just once
      _ss = iszero(j) ? ss : zero(Int8)
      duration = amt/rate
      @assert amt != zero(amt) || _ss == 1 || evid == 2
      if iszero(amt) && evid != 2
        @assert rate > zero(rate)
        # These are dose events having AMT=0, RATE>0, SS=1, and II=0.
        # Such an event consists of infusion with the stated rate,
        # starting at time −∞, and ending at the time on the dose
        # ev event record. Bioavailability fractions do not apply
        # to these doses.
        push!(events, Event(amt, t, evid, cmt, rate, ii, _ss, ii, t, Int8(1)))
      else
        push!(events, Event(amt, t, evid, cmt, rate, duration, _ss, ii, t, Int8(1)))
        if rate != zero(rate) && _ss == 0
          push!(events, Event(amt, t + duration, Int8(-1), cmt, rate, duration, _ss, ii, t, Int8(-1)))
        end
      end
      t += ii
    end
  end
  sort!(Vector{typeof(first(events))}(events))
end
