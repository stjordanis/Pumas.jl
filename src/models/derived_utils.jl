"""
    tad(t, events) -> time after most recent dose

Converts absolute time `t` (scalar or array) to relative time after the
most recent dose. If `t` is earlier than all dose times, then the (negative)
difference between `t` and the first dose time is returned instead. If no
dose events exist, `t` is returned unmodified.
"""
function tad(t::T, events::AbstractArray{E}) where {T<:Real, E<:Event}
  dose_times = [ev.time for ev in events if ev.evid == 1 || ev.evid == 4]
  isempty(dose_times) && return t
  sort!(dose_times)
  ind = searchsortedlast(dose_times, t)
  if ind > 0
    t - dose_times[ind]
  else
    t - dose_times[1]
  end
end
function tad(t::AbstractArray{T}, events::AbstractArray{E}) where {T<:Real, E<:Event}
  dose_times = [ev.time for ev in events if ev.evid == 1 || ev.evid == 4]
  isempty(dose_times) && return t
  sort!(dose_times)
  tout = similar(t)
  for i in eachindex(t)
    ind = searchsortedlast(dose_times, t[i])
    if ind > 0
      tout[i] = t[i] - dose_times[ind]
    else
      tout[i] = t[i] - dose_times[1]
    end
  end
  tout
end

"""
  eventnum(t, events) -> # of doses for each time

Creates an array that mataches `t` in length which denotes the number of events
that have occured prior to the current point in time. If `t` is a scalar, outputs
the number of events before that time point.
"""
function eventnum(t::T, events::AbstractArray{E}) where {T<:Real, E<:Event}
  findlast(ev->(ev.evid == 1 || ev.evid == 4) && ev.time < t ,events)
end
function eventnum(t::AbstractArray{T}, events::AbstractArray{E}) where {T<:Real, E<:Event}
  dose_times = [ev.time for ev in events if ev.evid == 1 || ev.evid == 4]
  [findlast(ev->(ev.evid == 1 || ev.evid == 4) && ev.time < _t ,events) for _t in t]
end
