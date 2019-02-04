struct SimulatedObservations{S,T,T2}
  subject::S
  times::T
  derived::T2
end

# indexing
@inline function Base.getindex(obs::SimulatedObservations, I...)
  return obs.derived[I...]
end
@inline function Base.setindex!(obs::SimulatedObservations, x, I...)
  obs.derived[I...] = x
end
function DataFrames.DataFrame(obs::SimulatedObservations)
  DataFrame(merge((time=obs.times,),obs.derived))
end

@recipe function f(obs::SimulatedObservations)
  t = obs.times
  names = Symbol[]
  plot_vars = []
  for (n,v) in pairs(obs.derived)
    push!(names,n)
    push!(plot_vars,v)
  end
  xlabel --> "time"
  ylabel --> reshape(names,1,length(names))
  layout --> good_layout(length(names))
  title --> "Subject ID: $(obs.subject.id)"
  t,plot_vars
end

function good_layout(n)
  n == 1 && return (1,1)
  n == 2 && return (2,1)
  n == 3 && return (3,1)
  n == 4 && return (2,2)
  n > 4 && return (n÷2,2)
end
