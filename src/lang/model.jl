"""
    PKPDModel

A model takes the following arguments
- `param`: a `ParamSet` detailing the parameters and their domain
- `data`: a `Population` object
- `random`: a mapping from a named tuple of parameters -> `RandomEffectSet`
- `pre`: a mapping from the (params, rfx, covars) -> ODE params
- `ode`: an ODE system (either exact or analytical)
- `derived`: the post processing (param, rfx, data, ode vals) -> outputs / sampling dist

The idea is that a user can then do
    fit(model, FOCE)
etc. which would return a FittedModel object

Note:
- we include the data in the model, since they are pretty tightly coupled

Todo:
- auxiliary mappings which don't affect the fitting (e.g. concentrations)
"""
mutable struct PKPDModel{P,Q,R,S,T,V}
    param::P
    random::Q
    pre::R
    init::S
    prob::T
    derived::V
    function PKPDModel(param, random, pre, init, ode, derived)
        prob = ode === nothing ? nothing : ode
        new{typeof(param), typeof(random),
            typeof(pre), typeof(init),
            Union{DiffEqBase.DEProblem,ExplicitModel,Nothing},
            typeof(derived)}(
            param, random, pre, init, prob, derived)
    end
end

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
                tspan::Tuple{Float64,Float64}=timespan(subject), kwargs...)
  if :saveat in keys(kwargs)
      tspan = min(tspan[1], first(kwargs[:saveat])), max(tspan[end], last(kwargs[:saveat]))
  end
  u0  = m.init(col, tspan[1])
  m.prob === nothing && return nothing
  if m.prob isa ExplicitModel
      return _solve_analytical(m, subject, u0, tspan, col, args...;kwargs...)
  else
      m.prob = remake(m.prob; p=col, u0=u0, tspan=tspan)
      return _solve_diffeq(m, subject, args...;kwargs...)
  end
end

"""
sample(d)

Samples a random value from a distribution or if it's a number assumes it's the
constant distribution and passes it through.
"""
sample(d::Distributions.Sampleable) = rand(d)
sample(d) = d

zval(d) = 0.0
zval(d::Distributions.Normal{T}) where {T} = zero(T)

"""
_lpdf(d,x)

The logpdf. Of a non-distribution it assumes the Dirac distribution.
"""
_lpdf(d,x) = d == x ? 0.0 : -Inf
_lpdf(d::Distributions.Sampleable,x) = logpdf(d,x)

function derivedfun(m::PKPDModel, subject::Subject,
                  param = init_param(m),
                  rfx=rand_random(m, param),
                  args...; continuity=:left,kwargs...)
  col = m.pre(param, rfx, subject.covariates)
  sol = _solve(m, subject, col, args...; kwargs...)
  derivedfun(m,col,sol;continuity=continuity)
end

function derivedfun(m::PKPDModel, col, sol; continuity=:left)
  if sol isa PKPDAnalyticalSolution
      derived = obstimes -> m.derived(col,sol.(obstimes),obstimes)
  elseif sol === nothing
      derived = obstimes -> m.derived(col,nothing,obstimes)
  else
      derived = obstimes -> m.derived(col,sol.(obstimes,continuity=continuity),obstimes)
  end
  derived
end

struct SimulatedObservations{T,T2}
    times::T
    derived::T2
end

# TODO: interface on SimulatedObservations
## size
#Base.length(A::SimulatedObservations) = length(A.obs)
#Base.size(A::SimulatedObservations) = size(A.obs)
#
# indexing
@inline function Base.getindex(A::SimulatedObservations, I...)
    return A.derived[I...]
end
@inline function Base.setindex!(A::SimulatedObservations, x, I...)
    A.derived[I...] = x
end
#Base.axes(A::SimulatedObservations) = axes(A.obs)
#Base.IndexStyle(::Type{<:SimulatedObservations}) = Base.IndexStyle(DataFrame)

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
                continuity=:left,
                obstimes=observationtimes(subject),kwargs...)
    col = m.pre(param, rfx, subject.covariates)
    if :saveat in keys(kwargs)
        isempty(kwargs[:saveat]) && throw(ArgumentError("saveat is empty."))
        sol = _solve(m, subject, col, args...; kwargs...)
    else
        isempty(obstimes) && throw(ArgumentError("obstimes is not specified."))
        sol = _solve(m, subject, col, args...; saveat=obstimes, kwargs...)
    end
    derived = derivedfun(m,col,sol;continuity=continuity)
    SimulatedObservations(obstimes,derived(obstimes)[1]) # the first component is observed values
end

function simobs(m::PKPDModel, pop::Population, args...;
                parallel_type = Threading, kwargs...)
    time = @elapsed if parallel_type == Serial
        sols = [simobs(m,subject,args...;kwargs...) for subject in pop]
    elseif parallel_type == Threading
        _sols = Vector{Any}(undef,length(pop))
        Threads.@threads for i in 1:length(pop)
            _sols[i] = simobs(m,pop[i],args...;kwargs...)
        end
        sols = [sol for sol in _sols] # Make strict typed
    elseif parallel_type == Distributed
        sols = pmap((subject)->simobs(m,subject,args...;kwargs...),pop)
    elseif parallel_type == SplitThreads
        error("SplitThreads is not yet implemented")
    end
    sols
end

"""
_likelihood(err, obs)

Computes the log-likelihood between the err and obs, only using err terms that
also have observations, and assuming the Dirac distribution for any err terms
that are numbers.
"""
function _likelihood(err::T, obs) where {T}
  syms =  fieldnames(T) ∩ fieldnames(typeof(obs.val))
  sum(map((d,x) -> isnan(x) ? zval(d) : _lpdf(d,x), (getproperty(err,x) for x in syms), (getproperty(obs.val,x) for x in syms)))
end

Base.@pure flattentype(t) = NamedTuple{fieldnames(typeof(t)), NTuple{length(t), eltype(eltype(t))}}

"""
    likelihood(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the full log-likelihood of model `m` for `subject` with parameters `param` and
random effects `rfx`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
function likelihood(m::PKPDModel, subject::Subject, args...; kwargs...)
   obstimes = [obs.time for obs in subject.observations]
   isempty(obstimes) && throw(ArgumentError("obstimes is not specified."))
   derived = derivedfun(m,subject,args...;kwargs...)
   _, derived_dist = derived(obstimes) # the second component is distributions
   typ = flattentype(derived_dist)
   n = length(derived_dist)
   idx = 1
   sum(subject.observations) do obs
       if eltype(derived_dist) <: Array
           l = _likelihood(typ(ntuple(i->derived_dist[i][idx], n)), obs)
       else
           l = _likelihood(derived_dist, obs)
       end
       idx += 1
       return l
   end
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
