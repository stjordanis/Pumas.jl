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

function do_ivivc(vitro_pop, ref_vivo_pop, vivo_pop, kwargs...)
  # model the vitro data
  # TODO: if vitro model is not specified by user then try all available models
  # and select the best one on the basis of aic, aicc and bic
  for i = 1:length(vitro_pop.subjects)
    for (key, value) in vitro_pop.subjects[i]
      vitro_model(vitro_pop.subjects[i][key], :emax)
    end
  end
  # model the reference vivo data and get required params ka, kel and V
  # if the formulation is "solution" then default model is bateman
  # TODO: if the formulation is "IV" then use exponential decaying model (not available currently)
  ka, kel, V = 0.0, 0.0, 0.0
  for i = 1:length(ref_vivo_pop.subjects)
    # TODO: if there are more than one ref data id then we hava to take avg of ka, kel and V over all ids??
    for (key, value) in ref_vivo_pop.subjects[i]
      if "solution" == lowercase(key)
        vivo_model(ref_vivo_pop.subjects[i][key], :bateman)
        ka, kel, V = ref_vivo_pop.subjects[i][key].pmin
      else
        error("not implemented yet!!")
      end
    end
  end
  # get the absorption profile from vivo data
  # TODO: add if-else for all kind of methods (model dependent, model independent and numerical deconvolution)
  # if none of the available methods are provided then we can run all methods and select the one which works
  # best with dissolution data (in vitro data) based on validation

  # for now, take deconvo_method as dummy function keyword (given by user)
  deconvo_method = :wn # wagner_nelson method
  vect = Vector{Dict{Any, Vector}}(undef, length(vivo_pop.subjects))
  for i = 1:length(vivo_pop.subjects)
    dict = Dict()
    sub = vivo_pop.subjects[i]
    for (key, value) in vivo_pop.subjects[i]
      dict[key] = get_fabs(sub[key].conc, sub[key].time, kel, deconvo_method)
    end
    vect[i] = dict
  end
  # now, correlate every individual to vitro data and establish a ivivc model
  # IVIVC models:
  #       1. Fabs(t) = AbsScale*Fdiss(t*Tscale)
  #       2. Fabs(t) = AbsScale*Fdiss(t*Tscale - Tshift)
  #       3. Fabs(t) = AbsScale*Fdiss(t*Tscale - Tshift) - AbsBase

  # take avg of fabs (all formulations) over all ids or correlate dissolution with each vivo ids
  # then take avg of ivivc model params??
  abs_scale, t_scale, c = 0.0, 0.0, 0
  avg_fabs = get_avg_fabs(vect)
  for (key, value) in avg_fabs
    # let's fit ivivc model 1.
    model_1(x, p) = @. p[1] * vitro_pop.subjects[1][key](x * p[2])
    p1, p2 = Curvefit(value, vivo_pop.subjects[1][key].time, model_1, 
                              [0.8, 0.6], LBFGS()).pmin
    abs_scale += p1; t_scale += p2
    c += 1
  end
  # take avg of abs_scale and t_scale
  abs_scale /= c; t_scale /= c
  do_ivivc{typeof(ka)}(vitro_pop, ref_vivo_pop, vivo_pop, ka, kel, V, vect, abs_scale, t_scale)
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
