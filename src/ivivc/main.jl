# main function for whole IVIVC pipeline

# do_ivivc(; vitro_data::AbstractString, ref_vivo_data::AbstractString,
#   vivo_data::AbstractString, kwargs...) = do_ivivc(read_vitro(vitro_data, kwargs...),
#                       read_vivo(ref_vivo_data, kwargs...), read_vivo(vivo_data, kwargs...),
#                       kwargs...)

struct do_ivivc{pType}
  vitro_pop::VitroPopulation
  ref_vivo_pop::VivoPopulation
  vivo_pop::VivoPopulation
  ka::pType
  kel::pType
  V::pType
  vect::Vector{Dict{Any, Vector}}
  AbsScale::pType
  Tscale::pType
  ### TODO: add other members
end

function do_ivivc(vitro_batch, ref_vivo_pop, vivo_pop;
                    vitro_m=nothing,
                    vitro_model_metric=:aic,
                    ref_vivo_model=nothing,
                    deconvo_method=:wn,
                    corr_m=:two)
  # model the vitro data
  # TODO: if vitro model is not specified by user then try all available models
  # and select the best one on the basis of aic, aicc and bic
  if vitro_m == nothing
    for (form, prof) in vitro_batch[1]
      try_all_vitro_model(prof, metric=vitro_model_metric)
    end
  else
    for (form, prof) in vitro_batch[1]
      vitro_model(prof, vitro_m)
    end
  end

  # model the reference vivo data and get required params ka, kel and V
  # if the formulation is "solution" then default model is bateman
  # TODO: if the formulation is "IV" then use exponential decaying model (not available currently)
  ka, kel, V = 0.0, 0.0, 0.0
  if ref_vivo_model == nothing
    ref_vivo_model = lowercase(first(keys(ref_vivo_pop[1]))) == "solution" ? (:bateman) : (:iv)
    @show ref_vivo_model
  end
  
  for (form, prof) in ref_vivo_pop[1]
    vivo_model(prof, ref_vivo_model)
    ka, kel, V = prof.pmin
  end

  # get the absorption profile from vivo data
  # TODO: add if-else for all kind of methods (model dependent, model independent and numerical deconvolution)
  # if none of the available methods are provided then we can run all methods and select the one which works
  # best with dissolution data (in vitro data) based on validation

  abs_vect = Vector{Dict{Any, Vector}}(undef, length(vivo_pop))
  for i = 1:length(vivo_pop)
    dict = Dict()
    for (form, prof) in vivo_pop[i]
      dict[form] = get_fabs(prof.conc, prof.time, kel, deconvo_method)
    end
    abs_vect[i] = dict
  end

  # now, correlate every individual to vitro data and establish a ivivc model
  # IVIVC models:
  #       1. Fabs(t) = AbsScale*Fdiss(t*Tscale)
  #       2. Fabs(t) = AbsScale*Fdiss(t*Tscale - Tshift)
  #       3. Fabs(t) = AbsScale*Fdiss(t*Tscale - Tshift) - AbsBase

  # take avg of fabs (all formulations) over all ids or correlate dissolution with each vivo ids
  # then take avg of ivivc model params??

  abs_scale, t_scale, c = 0.0, 0.0, 0
  avg_fabs = get_avg_fabs(abs_vect)
  for (form, fabs) in avg_fabs
    if corr_m == :two
      corr_model(x, p) = @. p[1] * vitro_batch[1][form](x * p[2])
      p1, p2 = Curvefit(fabs, vivo_pop[1][form].time, corr_model, 
                              [0.8, 0.6], LBFGS()).pmin
      abs_scale += p1; t_scale += p2
      c += 1
    end
  end
  # take avg of abs_scale and t_scale
  abs_scale /= c; t_scale /= c
  do_ivivc{typeof(ka)}(vitro_batch, ref_vivo_pop, vivo_pop, ka, kel, V, abs_vect, abs_scale, t_scale)
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
function get_avg_fabs(vect)
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
