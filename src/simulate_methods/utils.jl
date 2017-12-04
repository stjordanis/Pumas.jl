function adjust_event_timings(datai,lags,bioav,rate,duration)
  events = datai.events
  change_times!(events,lags,bioav,rate,duration)
  sort!(events)
  tstop_times = sorted_approx_unique(events)
  events,tstop_times
end

function set_value(A :: SVector{L,T}, x,k) where {T,L}
    SVector(ntuple(i->ifelse(i == k, x, A[i]), Val{L}))
end

function increment_value(A :: SVector{L,T}, x,k) where {T,L}
    SVector(ntuple(i->ifelse(i == k, A[i]+x, A[i]), Val{L}))
end

function increment_value(A::Number,x,k)
  A+x
end
