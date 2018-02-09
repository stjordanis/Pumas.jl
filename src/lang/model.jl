export PKPDModel, init_param, init_random, rand_random, pkpd_simulate, pkpd_likelihood, pkpd_post, pkpd_postfun

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
struct PKPDModel{P,Q,R,S,T,U}
    param::P
    random::Q
    collate::R
    ode::S
    post::T
    error::U
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
                       args...; kwargs...)
    col = m.collate(param, rfx, subject.covariates)
    sol = pkpd_solve(m.ode, subject, col, args...; kwargs...)
    return sol, col
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
    sol, col = pkpd_solve(m, subject, param, rfx, args...; saveat=obstimes, kwargs...)
    map(obstimes) do t
        # TODO: figure out a way to iterate directly over sol(t)
        errdist = m.error(param,rfx,subject.covariates,col,sol(t),t)
        map(rand, errdist)
    end        
end


"""
    pkpd_likelihood(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the full log-likelihood of model `m` for `subject` with parameters `param` and
random effects `rfx`. `args` and `kwargs` are passed to ODE solver.
"""
function pkpd_likelihood(m::PKPDModel, subject::Subject, param, rfx,
                         args...; kwargs...)
    obstimes = observationtimes(subject)
    sol, col = pkpd_solve(m, subject, param, rfx, args...; saveat=obstimes, kwargs...)
    sum(subject.observations) do obs
        # TODO: figure out a way to iterate directly over sol(t)
        t = obs.time
        errdist = m.error(param,rfx,subject.covariates,col,sol(t),t)
        sum(map(logpdf,errdist,obs.val))
    end
end


function pkpd_post(m::PKPDModel, subject::Subject, param,
                       rfx=rand_random(m, param),
                       args...; obstimes=observationtimes(subject),kwargs...)
    sol, col = pkpd_solve(m, subject, param, rfx, args...; saveat=obstimes, kwargs...)
    map(obstimes) do t
        # TODO: figure out a way to iterate directly over sol(t)
        m.post(param,rfx,subject.covariates,col,sol(t),t)
    end        
end

function pkpd_postfun(m::PKPDModel, subject::Subject, param,
                       rfx=rand_random(m, param),
                       args...; kwargs...)
    sol, col = pkpd_solve(m, subject, param, rfx, args...; kwargs...)
    function post(t)
        m.post(param,rfx,subject.covariates,col,sol(t),t)
    end
    return post
end
