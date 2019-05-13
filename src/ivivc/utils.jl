# In vitro data modeling
function vitro_model(subj::VitroSubject, model::Union{Symbol, Function}; time_lag=false,
                        p0=nothing, alg=LBFGS(), upper_bound=nothing, lower_bound=nothing)
    fill_p0_and_bounds!(subj.conc, subj.time, model, time_lag, p0, upper_bound, lower_bound)
    pmin = Curvefit(subj.conc, subj.time, model, p0, alg, box=false, lb=lower_bound, ub=upper_bound).pmin
    subj.m = model; subj.alg = alg; subj.p0 = p0; subj.ub = upper_bound; subj.lb = lower_bound
    subj.pmin = pmin
    subj
end

function fill_p0_and_bounds!(conc, time, model, time_lag, p0, ub, lb)
  if model <: Symbol
    avail_models = get_avail_models()
    if model in keys(avail_models)
      ind_filler()[model](conc, time, time_lag, p0, ub, lb)
    else
      error("given model is not available, please pass the functional form")
    end
  else
    if p0 == nothing || lb == nothing || ub == nothing
      error("for custom model, please provide initial values of model params, lower bound and upper bound!!")
    end
  end

function (subj::VitroSubject)(t::Union{AbstractVector{<:Number}, Number})
  subj.m <: Symbol ? get_avail_models()[subj.m](t, subj.pmin) : subj.m(t, subj.pmin)
end

get_avail_models() = Dict([(:emax, emax), (:emax_ng, emax_ng), (:weibull, weibull),
                          (:d_weibll, double_weibull), (:makoid, makoid), (:e, emax),
                          (:eng, emax_ng), (:w, weibull), (:dw, double_weibull),
                          (:m, makoid)])

ind_filler() = Dict([(:emax, _emax), (:emax_ng, _emax_ng), (:weibull, _weibull),
                          (:d_weibll, _double_weibull), (:makoid, _makoid), (:e, _emax),
                          (:eng, _emax_ng), (:w, _weibull), (:dw, _double_weibull),
                          (:m, _makoid)])

function _emax(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? zeros(eltype(conc), 4)
    ub = ub === nothing ? [125.0, Inf, time[end], time[end]]
    p0 = p0 === nothing ? [conc[end], 1.0, time[2], time[2]]
  else
    lb = lb === nothing ? zeros(eltype(conc), 3)
    ub = ub === nothing ? [125.0, Inf, time[end]]
    p0 = p0 === nothing ? [conc[end], 1.0, time[2]]
  end
end

function _emax_ng(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? zeros(eltype(conc), 3)
    ub = ub === nothing ? [125.0, Inf, time[end], time[end]]
    p0 = p0 === nothing ? [conc[end], 1.0, time[2], time[2]]
  else
    lb = lb === nothing ? zeros(eltype(conc), 2)
    ub = ub === nothing ? [125.0, time[end]]
    p0 = p0 === nothing ? [conc[end], time[2]]
  end
end

function _weibull(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? zeros(eltype(conc), 4)
    ub = ub === nothing ? [Inf, time[end], Inf, time[end]]
    p0 = p0 === nothing ? [1.0, time[2], 3.0, time[2]]
  else
    lb = lb === nothing ? zeros(eltype(conc), 3)
    ub = ub === nothing ? [Inf, time[end], Inf]
    p0 = p0 === nothing ? [1.0, 2.0, 3.0]
  end
end

function _double_weibull(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? zeros(eltype(conc), 7)
    ub = ub === nothing ? [125.0, 125.0, time[end], 25.0, time[end], 25.0, time[end]]
    p0 = p0 === nothing ? [40.0, 80.0, time[2], 1.0, time[2], 1.0, time[2]]
  else
    lb = lb === nothing ? zeros(eltype(conc), 6)
    ub = ub === nothing ? [125.0, 125.0, time[end], 25.0, time[end], 25.0]
    p0 = p0 === nothing ? [40.0, 80.0, 5.0, 1.0, 7.0, 1.0]
  end
end

function _makoid(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? zeros(eltype(conc), 4)
    ub = ub === nothing ? [Inf, Inf, Inf, time[end]]
    p0 = p0 === nothing ? [1.0, 2.0, 3.0, time[end]]
  else
    lb = lb === nothing ? zeros(eltype(conc), 3)
    ub = ub === nothing ? [Inf, Inf, Inf]
    p0 = p0 === nothing ? [1.0, 2.0, 3.0]
end
