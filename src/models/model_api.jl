"""
    PKPDModel

A model takes the following arguments
- `param`: a `ParamSet` detailing the parameters and their domain
- `data`: a `Population` object
- `random`: a mapping from a named tuple of parameters -> `DistSet`
- `pre`: a mapping from the (params, rfx, covars) -> ODE params
- `ode`: an ODE system (either exact or analytical)
- `derived`: the derived variables and error distributions (param, rfx, data, ode vals) -> sampling dist
- `observed`: simulated values from the error model and post processing: (param, rfx, data, ode vals, samples) -> vals

The idea is that a user can then do
    fit(model, FOCE)
etc. which would return a FittedModel object

Note:
- we include the data in the model, since they are pretty tightly coupled

Todo:
- auxiliary mappings which don't affect the fitting (e.g. concentrations)
"""
mutable struct PKPDModel{P,Q,R,S,T,V,W}
  param::P
  random::Q
  pre::R
  init::S
  prob::T
  derived::V
  observed::W
end
PKPDModel(param,random,pre,init,prob,derived) =
    PKPDModel(param,random,pre,init,prob,derived,(col,sol,obstimes,samples)->samples)

init_param(m::PKPDModel) = init(m.param)
init_random(m::PKPDModel, param) = init(m.random(param))

"""
    rand_random(m::PKPDModel, param)

Generate a random set of random effects for model `m`, using parameters `param`.
"""
rand_random(m::PKPDModel, param) = rand(m.random(param))


"""
    sol = pkpd_solve(m::PKPDModel, subject::Subject, param,
                     rfx=rand_random(m, param),
                     args...; kwargs...)

Compute the ODE for model `m`, with parameters `param` and random effects
`rfx`. `alg` and `kwargs` are passed to the ODE solver. If no `rfx` are
given, then they are generated according to the distribution determined
in the model.

Returns a tuple containing the ODE solution `sol` and collation `col`.
"""
function DiffEqBase.solve(m::PKPDModel, subject::Subject,
                          param = init_param(m),
                          rfx = rand_random(m, param),
                          args...; kwargs...)
  m.prob === nothing && return nothing
  col = m.pre(param, rfx, subject.covariates)
  _solve(m,subject,col,args...;kwargs...)
end

@enum ParallelType Serial=1 Threading=2 Distributed=3 SplitThreads=4
function DiffEqBase.solve(m::PKPDModel, pop::Population,
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
  MonteCarloSolution(sols,time,true)
end

"""
This internal function is just so that the collation doesn't need to
be repeated in the other API functions
"""
function _solve(m::PKPDModel, subject, col, args...;
                tspan=nothing, kwargs...)
  m.prob === nothing && return nothing
  if tspan === nothing
    _tspan = timespan(subject)
    if m.prob isa DiffEqBase.DEProblem && !(m.prob.tspan === (nothing, nothing))
      _tspan = (min(_tspan[1], m.prob.tspan[1]),
                max(_tspan[2], m.prob.tspan[2]))
    end
    tspan_tmp = :saveat in keys(kwargs) ? (min(_tspan[1], first(kwargs[:saveat])), max(_tspan[2], last(kwargs[:saveat]))) : _tspan
    tspan = float.(tspan_tmp)
  end
  u0  = m.init(col, tspan[1])
  if m.prob isa ExplicitModel
    return _solve_analytical(m, subject, u0, tspan, col, args...;kwargs...)
  else
    mtmp = PKPDModel(m.param,
                     m.random,
                     m.pre,
                     m.init,
                     remake(m.prob; p=col, u0=u0, tspan=tspan),
                     m.derived,
                     m.observed)
    return _solve_diffeq(mtmp, subject, args...;kwargs...)
  end
end

"""
sample(d)

Samples a random value from a distribution or if it's a number assumes it's the
constant distribution and passes it through.
"""
sample(d::Distributions.Sampleable) = rand(d)
sample(d::AbstractArray{<:Distributions.Sampleable}) = map(sample,d)
sample(d) = d


zval(d) = 0.0
zval(d::Distributions.Normal{T}) where {T} = zero(T)

function derivedfun(m::PKPDModel, subject::Subject,
                    param = init_param(m),
                    rfx=rand_random(m, param),
                    args...; continuity=:right,kwargs...)
  col = m.pre(param, rfx, subject.covariates)
  sol = _solve(m, subject, col, args...; kwargs...)
  derivedfun(m,col,sol;continuity=continuity)
end

function derivedfun(m::PKPDModel, col, sol; continuity=:left)
  if sol === nothing
    derived = obstimes -> m.derived(col,nothing,obstimes)
  else
    derived = obstimes -> m.derived(col,sol.(obstimes,continuity=continuity),obstimes)
  end
  derived
end

"""
    simobs(m::PKPDModel, subject::Subject, param[, rfx, [args...]];
                  obstimes=observationtimes(subject),kwargs...)

Simulate random observations from model `m` for `subject` with parameters `param` at
`obstimes` (by default, use the times of the existing observations for the subject). If no
`rfx` is provided, then random ones are generated according to the distribution
in the model.
"""
function simobs(m::PKPDModel, subject::Subject,
                param = init_param(m),
                rfx=rand_random(m, param),
                args...;
                continuity=:right,
                obstimes=observationtimes(subject),kwargs...)
  col = m.pre(param, rfx, subject.covariates)
  if :saveat in keys(kwargs)
    isempty(kwargs[:saveat]) && throw(ArgumentError("saveat is empty."))
    sol = _solve(m, subject, col, args...; kwargs...)
  else
    isempty(obstimes) && throw(ArgumentError("obstimes is not specified."))
    sol = _solve(m, subject, col, args...; saveat=obstimes, kwargs...)
  end
  derived = derivedfun(m,col,sol;continuity=continuity)(obstimes)
  obs = m.observed(col,sol,obstimes,map(PuMaS.sample,derived))
  SimulatedObservations(subject,obstimes,obs) # the first component is observed values
end

function simobs(m::PKPDModel, pop::Population, args...;
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
    pre(m::PKPDModel, subject::Subject, param, rfx)

Returns the parameters of the differential equation for a specific subject
subject to parameter and random effects choices. Intended for internal use
and debugging.
"""
function pre(m::PKPDModel, subject::Subject, param, rfx)
  m.pre(param, rfx, subject.covariates)
end

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
