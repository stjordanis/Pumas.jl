struct SimulatedObservations{S,T,TAD,T2}
  subject::S
  times::T
  time_after_doses::TAD
  obs::T2
  function SimulatedObservations(subject::S, times::T,
    obs::T2) where {S,T,T2}
    time_after_doses = tad(times, subject.events)
    new{S,T,typeof(time_after_doses),T2}(subject,times,time_after_doses,obs)
  end
end

# indexing
@inline function Base.getindex(obs::SimulatedObservations, I...)
  return obs.obs[I...]
end
@inline function Base.setindex!(obs::SimulatedObservations, x, I...)
  obs.obs[I...] = x
end
function DataFrames.DataFrame(obs::SimulatedObservations)
  DataFrame(merge((time=obs.times,
                   time_after_dose=obs.time_after_doses),
                  obs.obs))
end

@recipe function f(obs::SimulatedObservations)
  t = obs.times
  names = Symbol[]
  plot_vars = []
  for (n,v) in pairs(obs.obs)
    push!(names,n)
    push!(plot_vars,v)
  end
  xlabel --> "time"
  legend --> false
  lw --> 3
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

struct SimulatedPopulation{S}
  sims::S
end
@inline function Base.getindex(pop::SimulatedPopulation, I...)
  return pop.sims[I...]
end
@inline function Base.setindex!(pop::SimulatedPopulation, x, I...)
  pop.sims[I...] = x
end
function DataFrames.DataFrame(pop::SimulatedPopulation)
  dfs = [DataFrame(merge((id=[s.subject.id for i in 1:length(s.times)],
                          time=s.times, time_after_dose=s.time_after_doses),
                          s.obs)) for s in pop.sims]
  reduce(vcat,dfs)
end

@recipe function f(pop::SimulatedPopulation)
  for p in pop.sims
    @series begin
      lw --> 1.5
      title --> "Population Simulation"
      p
    end
  end
  nothing
end
