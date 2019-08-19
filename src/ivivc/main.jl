# main function for IVIVC pipeline

# do_ivivc(; vitro_data::AbstractString, ref_vivo_data::AbstractString,
#   vivo_data::AbstractString, kwargs...) = do_ivivc(read_vitro(vitro_data, kwargs...),
#                       read_vivo(ref_vivo_data, kwargs...), read_vivo(vivo_data, kwargs...),
#                       kwargs...)

struct do_ivivc{pType}
  vitro_data::VitroData
  uir_data::UirData
  vivo_data::VivoData
  ka::pType
  kel::pType
  V::pType
  vitro_model::Symbol
  deconvo_method::Symbol
  fabs::fabsType
  auc_inf::aucType
  # params which are needed for modeling
  ivivc_model::Function                 # model type
  opt_alg::Optim.FirstOrderOptimizer    # alg to optimize cost function
  p0::pType                             # intial values of params
  ub::Union{Nothing, pType}             # upper bound of params
  lb::Union{Nothing, pType}             # lower bound of params
  pmin::paramsType                      # optimized params
end

function do_ivivc(vitro_data, uir_data, vivo_data;
                    vitro_model=nothing,
                    vitro_model_metric=:aic,
                    uir_frac = 1.0
                    deconvo_method=:wn,
                    ivivc_model=:two)
  # model the vitro data
  if vitro_model === nothing
    error("Not implemented!!")
  else
    for data in vitro_data
      for (form, vitro_form) in data
        estimate_fdiss(vitro_form, vitro_model)
      end
    end
  end

  # model the reference vivo data and get required params ka, kel and V
  uir_model = lowercase(uir_data.form) == "solution" ? (:bateman) : (:iv)
  estimate_uir(uir_data, uir_model, frac = uir_frac)
  ka, kel, V = uir_data.pmin

  # get the absorption profile from vivo data
  all_fabs = Vector{Dict{Any, Vector}}(undef, length(vivo_data))
  all_auc_inf = Vector{Dict{Any, Any}}(undef, length(vivo_data))
  for i = 1:length(vivo_data)
    dict = Dict()
    _dict = Dict()
    for (form, prof) in vivo_data[i]
      dict[form], _dict[form] = get_fabs(prof.conc, prof.time, kel, deconvo_method)
    end
    all_fabs[i] = dict
    all_auc_inf[i] = _dict
  end

  # IVIVC models:
  #       1. Fabs(t) = AbsScale*Fdiss(t*Tscale)
  #       2. Fabs(t) = AbsScale*Fdiss(t*Tscale - Tshift)
  #       3. Fabs(t) = AbsScale*Fdiss(t*Tscale - Tshift) - AbsBase

  avg_fabs = avg_fabs(all_fabs)
  # optimization
  if ivivc_model == :two
    m(form, time, x) = x[1] * vitro_data[1][form](time * x[2])
  elseif ivivc_model == :three
    m(form, time, x) = x[1] * vitro_data[1][form](time * x[2] - x[3])
  elseif ivivc_model == :four
    m(form, time, x) = (x[1] * vitro_data[1][form](time * x[2] - x[3])) - x[4];
  else
    error("Incorrect keyword for IVIVC model!!")
  end
  function errfun(x)
    err = 0.0
    for (form, prof) in vivo_data[1]
      err = err + mse(m(form, prof.time, x), avg_fabs[form])
    end
    return err
  end
  # intial estimates of params and bounds
  p = [0.8, 0.5, 0.6, 0.6]
  ub = [1.25, 1.25, 1.25, 1.25]
  lb = [0.0, 0.0, 0.0, 0.0]
  opt_alg = LBFGS()
  od = OnceDifferentiable(p->errfun(p), p, autodiff=:finite)
  mfit = Optim.optimize(od, lb, ub, p, Fminbox(opt_alg))
  pmin = Optim.minimizer(mfit)
  do_ivivc{typeof(ka)}(vitro_data, uir_data, vivo_data, ka, kel, V, deconvo_method, all_fabs, all_auc_inf,
                          m, opt_alg, p, ub, lb, pmin)
end

# main function for prediction by estimated IVIVC model
function prediction(A::do_ivivc)
  @unpack 
  if(A.deconvo_method != :wn) error("Not implemented yet!!") end
  # ODE Formulation
  f(c, p, t) = kel * auc_inf(vivo_pop[1][form]) * pmin[1] * pmin[2] * r_diss(t * pmin[2], vitro_data[1][form].pmin) - kel * c

end

# helper function to call deconvo methods
function get_fabs(c, t, kel, method)
  if method == :wn
    return wagner_nelson(c, t, kel)
  else
    error("not implemented yet!!")
  end
end

# helper function to get avg of fabs profile over all ids for every formulation
function avg_fabs(vect)
  dict = Dict{Any, Vector}()
  for key in keys(vect[1])
    dict[key] = zero(vect[1][key])
  end
  for i = 1:length(vect)
    for (key, value) in vect[i]
      dict[key] .= dict[key] .+ value
    end
  end
  for (key, value) in dict
    dict[key] .= value ./ length(vect)
  end
  dict
end

# helper function to fit all available vitro models and select best one on the basis of AIC (by default)
function try_all_vitro_model(sub; metric=:aic)
  metric_funcs = get_metric_func()
  list_s = [:e, :w, :dw]
  metric_values = Float64[]
  for x in list_s
    vitro_model(sub, x)
    push!(metric_values, metric_funcs[metric](sub))
  end
  best_one = list_s[argmin(metric_values)]
  vitro_model(sub, best_one)
end

get_metric_func() = Dict([(:aic, aic), (:aicc, aicc), (:bic, bic)])
