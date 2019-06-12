_cmt_value(ev::Event, u0, var::Number, default) = var
function _cmt_value(ev::Event, u0, var::Union{Tuple,AbstractArray},default)
  ev.cmt ∈ keys(var) ? var[ev.cmt] : default
end
function _cmt_value(ev::Event, u0, var::Union{NamedTuple,SLArray,LArray},default)
  if ev.cmt isa Symbol
    ev.cmt ∈ keys(var) ? var[ev.cmt] : default
  else
    if ev.cmt <= length(u0)
      _cmt = keys(u0)[ev.cmt]
      _cmt ∈ keys(var) ? var[_cmt] : default
    else
      default
    end
  end
end

const DEFAULT_RATE = 0
const DEFAULT_DURATION = 0
const DEFAULT_BIOAV = 1
const DEFAULT_LAGS = 0

function _adjust_event(ev::Event,u0,lags,bioav,rate,duration)
  if ev.rate == -2
    if rate != DEFAULT_RATE
      _rate = rate
      _duration = ev.amt
      _duration /= _cmt_value(ev, u0, rate, DEFAULT_RATE)
      _duration *= _cmt_value(ev, u0, bioav, DEFAULT_BIOAV)
    elseif duration != DEFAULT_DURATION
      _duration = _cmt_value(ev, u0, duration, DEFAULT_DURATION)
      _rate = ev.amt/_duration
      _rate *= _cmt_value(ev, u0, bioav, DEFAULT_BIOAV)
    else
      throw(ArgumentError("either rate or duration should be set in the @pre block when the Event rate is -2"))
    end
  else # both are zero
    _rate = ev.rate
    _duration = ev.amt/ev.rate
    _duration *= _cmt_value(ev, u0, bioav, DEFAULT_BIOAV)
  end

  if ev.rate_dir == -1
    time = ev.base_time + _duration
  else
    time = ev.base_time
  end

  time += _cmt_value(ev, u0, lags, DEFAULT_LAGS)
  Event(ev.amt, time, ev.evid, ev.cmt, _rate, _duration, ev.ss, ev.ii, ev.base_time, ev.rate_dir)
end

"""
  adjust_event(event,lags,bioav,rate,duration) -> Event
  adjust_event(events,lags,bioav,rate,duration) -> Event[]

Adjust the event(s) with the "magic arguments" lags, bioav, rate
and duration. Returns either the original event or an adjusted event
if modified. If given a list of events, the output event list will
be sorted.
"""
function adjust_event(events::AbstractVector{<:Event},args...)
  out = collect(adjust_event(ev, args...) for ev in events)
  sort!(out)
end
function adjust_event(ev::Event,u0,lags,bioav,rate_input,duration_input)
  rate = _cmt_value(ev, u0, rate_input, DEFAULT_RATE)
  duration = _cmt_value(ev, u0, duration_input, DEFAULT_DURATION)
  @assert rate == DEFAULT_RATE || duration == DEFAULT_DURATION

  if ev.amt != 0
    _adjust_event(ev,u0,lags,bioav,rate,duration)
  else
    ev
  end
end

function sorted_approx_unique(events)
  tType = mapreduce(t -> typeof(t.time), promote_type, events)
  out = tType[events[1].time]
  for i in 2:length(events)
    if abs(out[end] - events[i].time) > 10eps(tType)*oneunit(tType)
      push!(out,events[i].time)
    end
  end
  out
end

# Ad-hoc fix for creating SLVectors from existing ones that differ
# only in one element, indexed by a Symbol
function Base.setindex(x::LabelledArrays.SLArray{S,T,N,L,Syms}, val::T,
  ind::Symbol) where {S,T,N,L,Syms}
  ind = findfirst(isequal(ind), Syms)
  Base.setindex(x, val, ind)
end

function set_value(A :: StaticVector{L,T}, x,k) where {T,L}
  typeof(A)(ntuple(i->ifelse(i == k, x, A[i]), Val(L)))
end
set_value(A::LabelledArrays.SLArray, x, k) = Base.setindex(A, x, k)

function increment_value(A :: StaticVector{L,T}, x,k) where {T,L}
  typeof(A)(ntuple(i->ifelse(i == k, A[i]+x, A[i]), Val(L)))
end
increment_value(A::LabelledArrays.SLArray, x, k) = Base.setindex(A,
  A[k] + x, k)

function increment_value(A::Number,x,k)
  A+x
end

function get_magic_args(p,u0,t0)
  if haskey(p,:lags)
    lags = p.lags
  else
    lags = zero(eltype(t0))
  end

  if haskey(p,:bioav)
    bioav = p.bioav
  else
    bioav = 1.0#one(eltype(u0))
  end

  if haskey(p,:rate)
    rate = p.rate
  else
    rate = 0.0#zero(eltype(u0))
  end

  if haskey(p,:duration)
    duration = p.duration
  else
    duration = 0.0#zero(eltype(u0))
  end

  lags,bioav,rate,duration
end



# promote to correct numerical type (e.g. to handle Duals correctly)
# TODO Unitful.jl support?
numtype(::Type{T}) where {T<:Number} = T
numtype(x::Number)        = typeof(x)
numtype(x::AbstractArray) = eltype(x)
numtype(x::Factorization) = eltype(x)
numtype(X::PDMats.AbstractPDMat) = numtype(eltype(X))
# numtype(x::Tuple)         = reduce(promote_type, map(numtype,x))
numtype(x::Tuple)         = promote_type(map(numtype, x)...)
numtype(x::NamedTuple) = numtype(values(x))
numtype(x::Function) = Float32 # To allow time-varying covariates
numtype(x::AbstractString) = Float32 # To allow string covariates

zero(x) = Base.zero(x)
zero(x::Tuple) = map(zero,x)
zero(x::NamedTuple) = map(zero,x)
