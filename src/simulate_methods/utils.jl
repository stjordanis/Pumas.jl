_cmt_value(ev::Event, var::Number) = var
_cmt_value(ev::Event, var) = var[ev.cmt]
function adjust_event_timings(events::AbstractVector{Event},args...)
  out = collect(adjust_event_timings(ev, args...) for ev in events)
  sort!(out)
end
function adjust_event_timings(ev::Event,lags,bioav,rate_input,duration_input)
  rate = _cmt_value(ev, rate_input)
  duration = _cmt_value(ev, duration_input)
  @assert rate == 0 || duration == 0

  if ev.amt != 0
    if rate != 0
      _rate = rate
      _duration = ev.amt
      _duration /= _cmt_value(ev, rate)
      _duration *= _cmt_value(ev, bioav)
    elseif duration != 0
      _duration = _cmt_value(ev, duration)
      _rate = ev.amt/_duration
      _rate *= _cmt_value(ev, bioav)
    else # both are zero
      _rate = ev.rate
      _duration = ev.amt/ev.rate
      _duration *= _cmt_value(ev, bioav)
    end

    if ev.rate_dir == -1
      time = ev.base_time + _duration
    else
      time = ev.base_time
    end

    time += _cmt_value(ev, lags)
    Event(ev.amt, time, ev.evid, ev.cmt, _rate, _duration, ev.ss, ev.ii,
          ev.base_time, ev.rate_dir)
  else
    ev # deep copy instead?
  end
end

function sorted_approx_unique(event)
  tType = typeof(first(event).time)
  out = Vector{typeof(first(event).time)}(undef, 1)
  out[1] = event[1].time
  for i in 2:length(event)
    if abs(out[end] - event[i].time) > 10eps(tType)
      push!(out,event[i].time)
    end
  end
  out
end

function set_value(A :: StaticVector{L,T}, x,k) where {T,L}
  SVector(ntuple(i->ifelse(i == k, x, A[i]), Val(L)))
end

function increment_value(A :: StaticVector{L,T}, x,k) where {T,L}
  SVector(ntuple(i->ifelse(i == k, A[i]+x, A[i]), Val(L)))
end

function increment_value(A::Number,x,k)
  A+x
end

function get_magic_args(p,u0,t0)
  if haskey(p,:lags)
    lags = p.lags
  else
    lags = 0.0#zero(eltype(t0))
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
numtype(X::PDMats.AbstractPDMat) = numtype(eltype(X))
numtype(x::Tuple)         = promote_type(map(numtype,x)...)
numtype(x::NamedTuple) = promote_type(map(numtype,x)...)
numtype(x::Function) = Float64 # To allow time-varying covars, could be better

zero(x) = Base.zero(x)
zero(x::Tuple) = map(zero,x)
zero(x::NamedTuple) = map(zero,x)
