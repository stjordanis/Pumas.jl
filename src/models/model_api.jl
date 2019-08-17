"""
    PumasModel

A model takes the following arguments
- `param`: a `ParamSet` detailing the parameters and their domain
- `random`: a mapping from a named tuple of parameters -> `DistSet`
- `pre`: a mapping from the (params, randeffs, subject) -> ODE params
- `init`: a mapping (col,t0) -> inital conditions
- `prob`: a DEProblem describing the dynamics (either exact or analytical)
- `derived`: the derived variables and error distributions (param, randeffs, data, ode vals) -> sampling dist
- `observed`: simulated values from the error model and post processing: (param, randeffs, data, ode vals, samples) -> vals
"""
mutable struct PumasModel{P,Q,R,S,T,V,W}
  param::P
  random::Q
  pre::R
  init::S
  prob::T
  derived::V
  observed::W
end
PumasModel(param,random,pre,init,prob,derived) =
    PumasModel(param,random,pre,init,prob,derived,(col,sol,obstimes,samples,subject)->samples)

init_param(m::PumasModel) = init(m.param)
init_randeffs(m::PumasModel, param) = init(m.random(param))

"""
    sample_randeffs(m::PumasModel, param)

Generate a random set of random effects for model `m`, using parameters `param`.
"""
sample_randeffs(m::PumasModel, param) = rand(m.random(param))

# How long to solve
function timespan(sub::Subject,tspan,saveat)
  if isempty(sub.events) && isempty(saveat) && isempty(sub.time) && tspan == (nothing,nothing)
    error("No timespan is given. This means no events, observations, or user chosen time span exist for the subject. Please check whether the data was input correctly.")
  end
  e_lo, e_hi = !isnothing(sub.events) && !isempty(sub.events) ? extrema(evt.time for evt in sub.events) : (Inf,-Inf)
  s_lo, s_hi = !isnothing(saveat) && !isempty(saveat) ? extrema(saveat) : (Inf,-Inf)
  obs_lo, obs_hi = !isnothing(sub.time) && !isempty(sub.time) ? extrema(sub.time) : (Inf,-Inf)
  lo = minimum((e_lo,s_lo,obs_lo))
  hi = maximum((e_hi,s_hi,obs_hi))
  tspan !== nothing && tspan[1] !== nothing && (lo = tspan[1]) # User override
  tspan !== nothing && tspan[2] !== nothing && (hi = tspan[2]) # User override
  lo == Inf && error("No starting time identified. Please supply events or obstimes")
  hi == -Inf && error("No ending time identified. Please supply events, obstimes")
  lo, hi
end

# Where to save
observationtimes(sub::Subject) = isnothing(sub.observations) &&
                                 !isnothing(sub.events) && !isempty(sub.events) ?
                                 (0.0:1.0:(sub.events[end].time+24.0)) :
                                 sub.time


"""
    sol = solve(m::PumasModel, subject::Subject, param,
                randeffs=sample_randeffs(m, param),
                saveat = observationtimes(subject),
                args...; kwargs...)

Compute the ODE for model `m`, with parameters `param`, random effects
`randeffs` and a collection of times to save the solution at `saveat`.
`args` and `kwargs` are passed to the ODE solver. If no `randeffs` are
given, then they are generated according to the distribution determined
in the model. If no `saveat` times are given, the times are chosen to be
the vector of observed times for `subject`.

Returns a tuple containing the ODE solution `sol` and collation `col`.
"""
function DiffEqBase.solve(m::PumasModel, subject::Subject,
                          param = init_param(m),
                          randeffs = sample_randeffs(m, param),
                          saveat = observationtimes(subject),
                          args...; kwargs...)
  m.prob === nothing && return nothing
  col = m.pre(param, randeffs, subject)
  _solve(m,subject,col,args...;saveat=saveat,kwargs...)
end

@enum ParallelType Serial=1 Threading=2 Distributed=3 SplitThreads=4
function DiffEqBase.solve(m::PumasModel, pop::Population,
                          param = init_param(m),
                          args...; parallel_type = Threading,
                          kwargs...)
  time = @elapsed if parallel_type == Serial
    sols = [solve(m,subject,param,args...;kwargs...) for subject in pop]
  elseif parallel_type == Threading
    _sols = Vector{Any}(undef,length(pop))
    Threads.@threads for i in 1:length(pop)
      _sols[i] = solve(m,pop[i],param,args...;kwargs...)
    end
    sols = [sol for sol in _sols] # Make strict typed
  elseif parallel_type == Distributed
    sols = pmap((subject)->solve(m,subject,param,args...;kwargs...),pop)
  elseif parallel_type == SplitThreads
    error("SplitThreads is not yet implemented")
  end
  EnsembleSolution(sols,time,true)
end

"""
This internal function is just so that the collation doesn't need to
be repeated in the other API functions
"""
function _solve(m::PumasModel, subject, col, args...;
                tspan=nothing, saveat=nothing, kwargs...)
  m.prob === nothing && return nothing
  if tspan === nothing
    tspan = float.(timespan(subject,tspan,saveat))
  end

  if m.prob isa ExplicitModel
    return _solve_analytical(m, subject, tspan, col, args...;kwargs...)
  elseif m.prob isa AnalyticalPKProblem
    pksol = _solve_analytical(m, subject, tspan, col, args...;kwargs...)
    _col = (col...,___pk=pksol)
    u0  = m.init(col, tspan[1])
    _prob = remake(m.prob.prob2; p=_col, u0=u0, tspan=tspan)
    numsol = solve(_prob,args...;saveat=saveat,alg=AutoTsit5(Rosenbrock23()),kwargs...)
    if saveat !== nothing
      t = saveat
      u = [[pksol(numsol.t[i]);numsol[i]] for i in 1:length(numsol)]
    else
      t = numsol.t
      u = numsol.u
    end
    return AnalyticalPKSolution(u,t,pksol,numsol)
  else
    u0  = m.init(col, tspan[1])
    mtmp = PumasModel(m.param,
                     m.random,
                     m.pre,
                     m.init,
                     remake(m.prob; p=col, u0=u0, tspan=tspan),
                     m.derived,
                     m.observed)
    return _solve_diffeq(mtmp, subject, args...;saveat=saveat, kwargs...)
  end
end

#=
_rand(d)

Samples a random value from a distribution or if it's a number assumes it's the
constant distribution and passes it through.
=#
_rand(d::Distributions.Sampleable) = rand(d)
_rand(d::AbstractArray{<:Distributions.Sampleable}) = map(_rand,d)
_rand(d) = d


zval(d) = 0.0
zval(d::Distributions.Normal{T}) where {T} = zero(T)

"""
    simobs(m::PumasModel, subject::Subject, param[, randeffs, [args...]];
                  obstimes=observationtimes(subject),kwargs...)

Simulate random observations from model `m` for `subject` with parameters `param` at
`obstimes` (by default, use the times of the existing observations for the subject). If no
`randeffs` is provided, then random ones are generated according to the distribution
in the model.
"""
function simobs(m::PumasModel, subject::Subject,
                param = init_param(m),
                randeffs=sample_randeffs(m, param),
                args...;
                obstimes=observationtimes(subject),
                saveat=obstimes,kwargs...)
  col = m.pre(param, randeffs, subject)
  m.prob !== nothing && (isnothing(obstimes) || isempty(obstimes)) &&
                          throw(ArgumentError("obstimes is empty."))
  sol = _solve(m, subject, col, args...; saveat=saveat, kwargs...)
  derived = m.derived(col,sol,obstimes,subject)
  obs = m.observed(col,sol,obstimes,map(_rand,derived),subject)
  SimulatedObservations(subject,obstimes,obs)
end

function simobs(m::PumasModel, pop::Population, args...;
                parallel_type = Threading, kwargs...)
  time = @elapsed if parallel_type == Serial
    sims = [simobs(m,subject,args...;kwargs...) for subject in pop]
  elseif parallel_type == Threading
    _sims = Vector{Any}(undef,length(pop))
    Threads.@threads for i in 1:length(pop)
      _sims[i] = simobs(m,pop[i],args...;kwargs...)
    end
    sims = [sim for sim in _sims] # Make strict typed
  elseif parallel_type == Distributed
    sims = pmap((subject)->simobs(m,subject,args...;kwargs...),pop)
  elseif parallel_type == SplitThreads
    error("SplitThreads is not yet implemented")
  end
  SimulatedPopulation(sims)
end

"""
    pre(m::PumasModel, subject::Subject, param, randeffs)

Returns the parameters of the differential equation for a specific subject
subject to parameter and random effects choices. Intended for internal use
and debugging.
"""
function pre(m::PumasModel, subject::Subject, param, randeffs)
  m.pre(param, randeffs, subject)
end
