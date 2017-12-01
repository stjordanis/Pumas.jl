function adjust_event_timings(datai,p,bioav)
  if !haskey(p,:lags)
    target_time = datai.event_times
    change_duration_by_bioav!(target_time,bioav)
    events = datai.events
    if bioav != 1
      order = sortperm(target_time)
      permute!(target_time,order)
      permute!(events,order)
    end
  else
    target_time,events = remove_lags(datai.events,datai.event_times,p.lags,bioav)
  end
  tstop_times = sorted_approx_unique(target_time)
  target_time,events,tstop_times
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
