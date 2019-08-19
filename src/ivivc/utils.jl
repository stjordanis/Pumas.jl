# In vitro data modeling
function estimate_fdiss(subj::VitroForm, model::Union{Symbol, Function}; time_lag=false,
                        p0=nothing, alg=LBFGS(), box=true, upper_bound=nothing, lower_bound=nothing)
    lower_bound, upper_bound, p0 = fill_p0_and_bounds(subj.conc, subj.time, model, time_lag, p0, upper_bound, lower_bound)
    model = typeof(model) <: Symbol ? get_avail_models()[model] : model
    pmin = Curvefit(subj.conc, subj.time, model, p0, alg, box, lower_bound, upper_bound).pmin
    subj.m = model; subj.alg = alg; subj.p0 = p0; subj.ub = upper_bound; subj.lb = lower_bound
    subj.pmin = pmin
    subj
end

# Estimate UIR
function estimate_uir(subj::UirData, model::Union{Symbol, Function}; frac=nothing, p0=nothing, alg=LBFGS(), 
                      box=false, upper_bound=nothing, lower_bound=nothing)
  lower_bound, upper_bound, p0 = _fill_p0_and_bounds(subj.conc, subj.time, model, p0, box, upper_bound, lower_bound)
  model = typeof(model) <: Symbol ? get_avail_uir_models()[model] : model
  m(t, p) = model(t, p) * frac * subj.dose  ### p[1] => ka, p[2] => kel, p[3] => V
  pmin = Curvefit(subj.conc, subj.time, m, p0, alg, box, lower_bound, upper_bound).pmin
  subj.m = m; subj.alg = alg; subj.p0 = p0; subj.ub = upper_bound; subj.lb = lower_bound
  subj.pmin = pmin
  subj
end

function fill_p0_and_bounds(conc, time, model, time_lag, p0, ub, lb)
  if typeof(model) <: Symbol
    avail_models = get_avail_models()
    if model in keys(avail_models)
      return ind_filler()[model](conc, time, time_lag, p0, ub, lb)
    else
      error("given model is not available, please pass the functional form")
    end
  else
    if p0 == nothing || lb == nothing || ub == nothing
      error("for custom model, please provide initial values of model params, lower bound and upper bound!!")
    end
  end
end

function _fill_p0_and_bounds(conc, time, model, p0, box, ub, lb)
  if typeof(model) <: Symbol
    avail_models = get_avail_uir_models()
    if model in keys(avail_models)
      return vivo_ind_filler()[model](conc, time, box, p0, ub, lb)
    else
      error("given model is not available, please pass the functional form")
    end
  else
    if p0 == nothing || lb == nothing || ub == nothing
      error("for custom model, please provide initial values of model params lower bound and upper bound!!")
    end
  end
end

function (subj::VitroForm)(t::Union{AbstractVector{<:Number}, Number})
  subj.m(t, subj.pmin)
end

function (subj::UirData)(t::Union{AbstractVector{<:Number}, Number})
  subj.m(t, subj.pmin)
end

# simpe MSE
mse(x, y) = sum(abs2.(x .- y))/length(x)

# simple linear regression function
linreg(x, y) = hcat(fill!(similar(x), 1), x) \ y # ans[1] => intercept, ans[2] => slope

get_avail_models() = Dict([(:emax, emax), (:emax_ng, emax_ng), (:weibull, weibull),
                          (:d_weibll, double_weibull), (:makoid, makoid), (:e, emax),
                          (:eng, emax_ng), (:w, weibull), (:dw, double_weibull),
                          (:m, makoid)])

get_avail_uir_models() = Dict([(:bateman, bateman), (:bat, bateman), (:iv, iv)])

ind_filler() = Dict([(:emax, _emax), (:emax_ng, _emax_ng), (:weibull, _weibull),
                          (:d_weibll, _double_weibull), (:makoid, _makoid), (:e, _emax),
                          (:eng, _emax_ng), (:w, _weibull), (:dw, _double_weibull),
                          (:m, _makoid)])

vivo_ind_filler() = Dict([(:bateman, _bateman), (:bat, _bateman), (:iv, _iv)])

function _emax(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? [0.0, 1.0, time[2], time[2]] : lb
    ub = ub === nothing ? [1.25, Inf, time[end], time[end]] : ub
    p0 = p0 === nothing ? [conc[end], 1.2, time[2], time[2]] : p0
  else
    lb = lb === nothing ? [0.0, 1.0, 0.0] : lb
    ub = ub === nothing ? [1.25, Inf, time[end]] : ub
    p0 = p0 === nothing ? [conc[end], 1.2, time[2]] : p0
  end
  lb, ub, p0
end

function _emax_ng(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? [0.0, time[2], time[2]] : lb
    ub = ub === nothing ? [1.25, time[end], time[end]] : ub
    p0 = p0 === nothing ? [conc[end], time[2], time[2]] : p0
  else
    lb = lb === nothing ? [0.0, time[2]] : lb
    ub = ub === nothing ? [1.25, time[end]] : ub
    p0 = p0 === nothing ? [conc[end], time[2]] : p0
  end
  lb, ub, p0
end

function _weibull(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? [0.0, 0.0, 1.0, 0.0] : lb
    ub = ub === nothing ? [1.25, time[end], Inf, time[end]] : ub
    p0 = p0 === nothing ? [conc[end], time[2], 1.2, time[2]] : p0
  else
    lb = lb === nothing ? [0.0, 0.0, 1.0] : lb
    ub = ub === nothing ? [1.25, time[end], Inf] : ub
    p0 = p0 === nothing ? [conc[end], time[2], 1.2] : p0
  end
  lb, ub, p0
end

function _double_weibull(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0] : lb
    ub = ub === nothing ? [1.25, 1.25, time[end], Inf, time[end], Inf, time[end]] : ub
    p0 = p0 === nothing ? [conc[end], conc[end], time[2], 1.2, time[2], 1.2] : p0
  else
    lb = lb === nothing ? [0.0, 0.0, 0.0, 1.0, 0.0, 1.0] : lb
    ub = ub === nothing ? [1.25, 1.25, time[end], Inf, time[end], Inf] : ub
    p0 = p0 === nothing ? [conc[end], conc[end], time[2], 1.2, time[2], 1.2] : p0
  end
  lb, ub, p0
end

function _makoid(conc, time, time_lag, p0, ub, lb)
  if time_lag
    lb = lb === nothing ? [0.0, 0.0, 1.0, 0.0] : lb
    ub = ub === nothing ? [1.25, 1.25, Inf, time[end]] : ub
    p0 = p0 === nothing ? [conc[end], conc[end], 1.2, time[end]] : p0
  else
    lb = lb === nothing ? [0.0, 0.0, 1.0] : lb
    ub = ub === nothing ? [1.25, time[end], Inf] : ub
    p0 = p0 === nothing ? [conc[end], time[2], 1.2] : p0
  end
  lb, ub, p0
end

function _bateman(conc, time, box, p0, ub, lb)
  if box
    error("TODO")
  else
    p0 = p0 === nothing ? [2.0, 3.0, 4.0] : p0  ## for now, initial estimation can be done using terminal slope (kel) and feathering methos (ka)
    ub = nothing
    lb = nothing
  end
  lb, ub, p0
end
