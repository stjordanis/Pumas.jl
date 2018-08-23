export PKPDModel, init_param, init_random, rand_random, pkpd_solve, pkpd_simulate, pkpd_likelihood, pkpd_post, pkpd_postfun

"""
    PKPDModel

A model takes the following arguments
- `param`: a `ParamSet` detailing the parameters and their domain
- `data`: a `Population` object
- `random`: a mapping from a named tuple of parameters -> `RandomEffectSet`
- `collate`: a mapping from the (params, rfx, covars) -> ODE params
- `ode`: an ODE system (either exact or analytical)
- `error`: the error model mapping (param, rfx, data, ode vals) -> sampling dist

The idea is that a user can then do
    fit(model, FOCE)
etc. which would return a FittedModel object

Note:
- we include the data in the model, since they are pretty tightly coupled

Todo:
- auxiliary mappings which don't affect the fitting (e.g. concentrations)
"""
mutable struct PKPDModel{P,Q,R,S,T,U,V}
    param::P
    random::Q
    collate::R
    init::S
    prob::T
    post::U
    error::V
    function PKPDModel(param, random, collate, init, ode, post, error)
        prob = ODEProblem(ODEFunction(ode), nothing, nothing, nothing)
        new{typeof(param), typeof(random),
            typeof(collate), typeof(init),
            DiffEqBase.DEProblem, typeof(post),
            typeof(error)}(param, random, collate, init, prob, post, error)
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
    (sol, col) = pkpd_solve(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the ODE for model `m`, with parameters `param` and random effects
`rfx`. `alg` and `kwargs` are passed to the ODE solver.

Returns a tuple containing the ODE solution `sol` and collation `col`.
"""
function pkpd_solve(m::PKPDModel, subject::Subject, param, rfx,
                    args...; tspan::Tuple{Float64,Float64}=timespan(subject), kwargs...)
    col = m.collate(param, rfx, subject.covariates)
    u0  = m.init(col, tspan[1])
    m.prob = remake(m.prob; p=col, u0=u0, tspan=tspan)

    sol = _solve(m, subject, args...;kwargs...)
    return sol, col
end

function _solve(m::PKPDModel, subject, args...;kwargs...)
    if m.prob.f.f isa ExplicitModel
        return _solve_analytical(m, subject, args...;kwargs...)
    else
        return _solve_diffeq(m, subject, args...;kwargs...)
    end
end

"""
    pkpd_simulate(m::PKPDModel, subject::Subject, param[, rfx, [args...]];
                  obstimes=observationtimes(subject),kwargs...)

Simulate random observations from model `m` for `subject` with parameters `param` at
`obstimes` (by default, use the times of the existing observations for the subject). If no
`rfx` is provided, then random ones are generated.
"""
function pkpd_simulate(m::PKPDModel, subject::Subject, param,
                       rfx=rand_random(m, param),
                       args...; obstimes=observationtimes(subject),kwargs...)
    sol, col = pkpd_solve(m, subject, param, rfx, args...; kwargs...)
    map(obstimes) do t
        # TODO: figure out a way to iterate directly over sol(t)
        errdist = m.error(col,sol(t),t)
        map(rand, errdist)
    end
end


function pkpd_map(f, m::PKPDModel, subject::Subject, param, rfx,
                         args...; kwargs...)
    obstimes = observationtimes(subject)
    sol, col = pkpd_solve(m, subject, param, rfx, args...; kwargs...)
    sum(subject.observations) do obs
        t = obs.time
        err = m.error(col,sol(t),t)
        f(err, obs)
    end
end


function pkpd_sum(f, m::PKPDModel, subject::Subject, param, rfx,
                         args...; kwargs...)
    obstimes = observationtimes(subject)
    sol, col = pkpd_solve(m, subject, param, rfx, args...; kwargs...)
    sum(subject.observations) do obs
        t = obs.time
        err = m.error(col,sol(t),t)
        f(err, obs)
    end
end

likelihood(err, obs) = sum(map((d,x) -> isnan(x) ? zval(d) : logpdf(d,x), err, obs.val))
zval(d) = 0.0
zval(d::Distributions.Normal{T}) where {T} = zero(T)


"""
    pkpd_likelihood(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the full log-likelihood of model `m` for `subject` with parameters `param` and
random effects `rfx`. `args` and `kwargs` are passed to ODE solver.
"""
function pkpd_likelihood(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)
    pkpd_sum(likelihood, m, subject, param, rfx, args...; kwargs...)
end





function pkpd_post(m::PKPDModel, subject::Subject, param,
                       rfx=rand_random(m, param),
                       args...; obstimes=observationtimes(subject),kwargs...)
    sol, col = pkpd_solve(m, subject, param, rfx, args...; kwargs...)
    map(obstimes) do t
        # TODO: figure out a way to iterate directly over sol(t)
        m.post(col,sol(t),t)
    end
end

function pkpd_postfun(m::PKPDModel, subject::Subject, param,
                       rfx=rand_random(m, param),
                       args...; kwargs...)
    sol, col = pkpd_solve(m, subject, param, rfx, args...; kwargs...)
    function post(t)
        m.post(col,sol(t),t)
    end
    return post
end
