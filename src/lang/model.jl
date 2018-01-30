export NLModel, pkpd_simulate, pkpd_likelihood

"""
    NLModel

A model takes the following arguments
- `param`: a `ParamSet` detailing the parameters and their domain
- `data`: a `PKData` object
- `random`: a mapping paramters to random effects
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
struct NLModel{P,Q,R,S,T}
    param::P
    random::Q
    collate::R
    ode::S
    error::T
end

function pkpd_simulate(m::NLModel,subject::Subject,param,rfx,alg=Tsit5();
                       kwargs...)
    p = m.collate(param, rfx, subject.covariates)
    pkpd_solve(m.ode, subject, p, alg; kwargs...)
end


function pkpd_likelihood(m::NLModel,subject::Subject,param,rfx,alg=Tsit5();
                         kwargs...)
    ϵ = sqrt(eps())
    p = m.collate(param, rfx, subject.covariates)
    sol = pkpd_solve(m.ode, subject, p, alg; kwargs...)
    sum(subject.observations) do obs
        t = obs.time
        dists = m.error(param,rfx,subject.covariates,p,sol(t+ϵ),t)
        sum(map(logpdf,dists,obs.val))
    end
end
