#Structures to store the quantiles of the simulations per dv, stratification and quantile
struct VPC_QUANT
  Fiftieth::AbstractVector
  Fifth_Ninetyfifth::AbstractVector
  Simulation_Percentiles::AbstractVector
end

struct OBS_QUANT
  Observation_Quantiles::AbstractVector
end

struct VPC_STRAT
  vpc_quant::Vector{VPC_QUANT}
  strat::Union{Symbol, Nothing}
  strata_val::AbstractVector
end

struct VPC_DV
  vpc_strat::Vector{VPC_STRAT}
  dv::Union{Symbol, Nothing}
end

struct OBS_VPC
  t::AbstractVector
  Obs_vpc::AbstractVector  
end

struct VPC
  vpc_dv::Vector{VPC_DV}
	Obs_vpc::OBS_VPC
  Simulations::AbstractVector
  idv::Symbol
  data::Population
end

#Compute the quantiles of the stratification covariate
function get_strat(data::Population, stratify_on::Symbol)
  strat_vals = []
  cov_vals = Float64[]
  for i in 1:length(data)
    push!(cov_vals, getproperty(data[i].covariates,stratify_on))
  end
  if length(unique(cov_vals)) <= 4
    return unique(cov_vals)
  else
    return quantile(cov_vals, [0.25,0.5,0.75,1.0])
  end
end

#Compute quantiles of the simulations for the population for a dv, idv and strata
function get_simulation_quantiles(sims::AbstractVector, reps::Integer, dv_::Symbol, idv_::AbstractVector, quantiles::AbstractVector, strat_quant::AbstractVector, stratify_on::Union{Symbol,Nothing})
  pop_quantiles = []
  for i in 1:reps
    sim = sims[i]
    quantiles_sim = []
    for t in 1:length(idv_)
      sims_t = [Float64[] for i in 1:length(strat_quant)]
      for j in 1:length(sim.sims)
        for strt in 1:length(strat_quant)
          if  stratify_on == nothing || (length(strat_quant)<4 && stratify_on != nothing && sim.sims[j].subject.covariates[stratify_on] <= strat_quant[strt])
            push!(sims_t[strt], getproperty(sim[j].observed,dv_)[t])
          elseif stratify_on != nothing && sim.sims[j].subject.covariates[stratify_on] <= strat_quant[strt]
            if strt > 1 && sim.sims[j].subject.covariates[stratify_on] > strat_quant[strt-1]
              push!(sims_t[strt], getproperty(sim[j].observed,dv_)[t])
            elseif strt == 1
              push!(sims_t[strt], getproperty(sim[j].observed,dv_)[t])
            end
          end
        end
      end
      push!(quantiles_sim, [quantile(sims_t[strt],quantiles) for strt in 1:length(strat_quant)])
    end 
    push!(pop_quantiles,quantiles_sim)
  end
  pop_quantiles
end

function get_observation_quantiles(data::Population, dv_::Symbol, idv_::AbstractVector, quantiles::AbstractVector, strat_quant::AbstractVector, stratify_on::Union{Symbol,Nothing})
  quantiles_obs = []
  obs_t = []
  for strt in 1:length(strat_quant)
    push!(obs_t, [Float64[] for i in 1:length(idv_)])
    for t in 1:length(idv_)
      for j in 1:length(data)
        if (stratify_on == nothing || (length(strat_quant)<4 && stratify_on != nothing && data[j].covariates[stratify_on] <= strat_quant[strt]))
          push!(obs_t[strt][t], getproperty(data[j].observations,dv_)[t])
        elseif stratify_on != nothing && data[j].covariates[stratify_on] <= strat_quant[strt]
          if strt > 1 && data[j].covariates[stratify_on] > strat_quant[strt-1]
            push!(obs_t[strt][t], getproperty(data[j].observations,dv_)[t])
          elseif strt == 1
            push!(obs_t[strt][t], getproperty(data[j].observations,dv_)[t])
          end
        end
      end
    end
    push!(quantiles_obs, OBS_QUANT([quantile(obs_t[strt][t],quantiles[2]) for t in 1:length(idv_)]))
  end 
  quantiles_obs
end

#Compute quantiles of the quantiles to get the values for the ribbons 
function get_quant_quantiles(pop_quantiles::AbstractVector, reps::Int, idv_::AbstractVector, quantiles::AbstractVector, strat_quant::AbstractVector)
  quantile_quantiles = []
  for strt in 1:length(strat_quant)
    quantile_strat = []
    for t in 1:length(idv_)
      quantile_time = []
      for j in 1:length(pop_quantiles[1][t][1])
        quantile_index = Float64[]
        for i in 1:reps
          push!(quantile_index,pop_quantiles[i][t][strt][j])
        end
        push!(quantile_time, quantile(quantile_index,quantiles))
      end
      push!(quantile_strat,quantile_time)
    end
    push!(quantile_quantiles,quantile_strat)
  end
  quantile_quantiles
end

#For each strata store it's quantiles of the quantiles
function get_vpc(quantile_quantiles::AbstractVector, strat_quant::AbstractVector, stratify_on::Union{Symbol,Nothing})
  vpc_strat = VPC_QUANT[]
  for strt in 1:length(strat_quant)
    fifty_percentiles = []
    fith_ninetyfifth = []
    for i in 1:3
      push!(fifty_percentiles,[j[i][2] for j in quantile_quantiles[strt]])
      push!(fith_ninetyfifth, [(j[i][1],j[i][3]) for j in quantile_quantiles[strt]])
    end
    push!(vpc_strat, VPC_QUANT(fifty_percentiles, fith_ninetyfifth, quantile_quantiles))
  end
  VPC_STRAT(vpc_strat, stratify_on, strat_quant)
end

"""
vpc(m::PumasModel, data::Population, fixeffs::NamedTuple, reps::Integer;quantiles = [0.05,0.5,0.95], idv = :time, dv = [:dv], stratify_on = nothing)

Computes the quantiles for VPC. The default quantiles are the 5th, 50th and 95th percentiles. 
  args: PumasModel, Population, Parameters and Number of Repetitions  
        
  Instead of the model, simulations from a previous vpc run (obtained from VPC.Simulations) or a FittedPumasModel can be used.

  kwargs: quantiles - Takes an array of quantiles to be calculated. The first three indices are used for plotting. 
          idv - The idv to be used, defaults to time. 
          dv - Takes an array of symbols of the dvs for which the quantiles are computed.
          stratify_on - Takes an array of symbols of covariates which the VPC is stratified on.
"""
function vpc(m::PumasModel, data::Population, fixeffs::NamedTuple, reps::Integer;quantiles = [0.05,0.5,0.95], idv = :time, dv = [:dv], stratify_on = nothing)
  # rand_seed = rand()
  # Random.seed!(rand_seed)
  # println("Seed set as $rand_seed")
  vpcs = VPC_DV[]
  obs_vpc = []
  strat_quants = []
  if stratify_on != nothing
    for strt in stratify_on
      strat_quant = get_strat(data, strt)
      push!(strat_quants , strat_quant)
    end
  else
    push!(strat_quants, [1])
  end

  if idv == :time
    idv_ = getproperty(data[1], idv)
  else 
    idv_ = [getproperty(data[i].covariates, idv) for i in 1:length(data)]
  end

  sims = [simobs(m, data, fixeffs) for i in 1:reps]

  if idv_ == nothing && idv == :time
    idv_ = Array(getproperty(sims[1][1], :times))
  end

  for dv_ in dv
    stratified_vpc = VPC_STRAT[]
    obs_vpc_dv = [] 
    for strat in 1:length(strat_quants)

      if stratify_on != nothing
        pop_quantiles = get_simulation_quantiles(sims, reps, dv_, idv_, quantiles, strat_quants[strat],stratify_on[strat])
        quantile_quantiles = get_quant_quantiles(pop_quantiles,reps,idv_,quantiles, strat_quants[strat])
        vpc_strat = get_vpc(quantile_quantiles, strat_quants[strat], stratify_on[strat])
        if data[1].observations != nothing
          obs_quantiles = get_observation_quantiles(data, dv_, idv_, quantiles, strat_quants[strat], stratify_on[strat])
        else
          obs_quantiles = [nothing for i in 1:length(strat_quants[strat])]
        end
      else
        pop_quantiles = get_simulation_quantiles(sims, reps, dv_, idv_, quantiles, strat_quants[strat],nothing)
        quantile_quantiles = get_quant_quantiles(pop_quantiles,reps,idv_,quantiles, strat_quants[strat])
        vpc_strat = get_vpc(quantile_quantiles, strat_quants[strat], nothing)
        if data[1].observations != nothing
          obs_quantiles = get_observation_quantiles(data, dv_, idv_, quantiles, strat_quants[strat], nothing)
        else
          obs_quantiles = [nothing for i in 1:length(strat_quants[strat])]
        end
      end

      push!(obs_vpc_dv, obs_quantiles)
      push!(stratified_vpc, vpc_strat)
    end
    push!(vpcs, VPC_DV(stratified_vpc, dv_))
    push!(obs_vpc, obs_vpc_dv) 
  end
  VPC(vpcs, OBS_VPC(idv_, obs_vpc), sims, idv, data)
end

function vpc_obs(data::Population;quantiles = [0.05,0.5,0.95], idv = :time, dv = [:dv], stratify_on = nothing)
  strat_quants = []
  if stratify_on != nothing
    for strt in stratify_on
      strat_quant = get_strat(data, strt)
      push!(strat_quants , strat_quant)
    end
  else
    push!(strat_quants, [1])
  end
  
  if idv == :time
    idv_ = getproperty(data[1], idv)
  else 
    idv_ = getproperty(data[1].covariates, idv)
  end
  
  obs_vpc = []
  for dv_ in dv
    obs_vpc_dv = []
    for strat in 1:length(strat_quants)
      if stratify_on != nothing
        push!(obs_vpc_dv, get_observation_quantiles(data, dv_, idv_, quantiles, strat_quants[strat], stratify_on[strat]))
      else
        push!(obs_vpc_dv, get_observation_quantiles(data, dv_, idv_, quantiles, strat_quants[strat], nothing))
      end
    end
    push!(obs_vpc, obs_vpc_dv)
  end
  OBS_VPC(idv_, obs_vpc)
end

#Use FittedPumasModel object for vpc
function vpc(fpm::FittedPumasModel, reps::Integer, data::Population=fpm.data;quantiles = [0.05,0.5,0.95], idv = :time, dv = [:dv], stratify_on = nothing)
  vpc(fpm.model, fpm.data, fpm.param, reps, quantiles=quantiles, idv=idv, dv=dv, stratify_on=stratify_on)
end

#Use simulations from a previous vpc calculation for a different statification
function vpc(sims::AbstractVector, data::Population;quantiles = [0.05,0.5,0.95], idv = :time, dv = [:dv], stratify_on = nothing)
  # rand_seed = rand()
  # Random.seed!(rand_seed)
  # println("Seed set as $rand_seed")
  if typeof(sims) == VPC
    sims = sims.Simulations
  end

  if idv == :time
    idv_ = getproperty(data[1], idv)
  else 
    idv_ = getproperty(data[1].covariates, idv)
  end

  vpcs = []
  obs_vpc = []
  strat_quants = []
  if stratify_on != nothing
    for strt in stratify_on
      strat_quant = get_strat(data, strt)
      push!(strat_quants, strat_quant)
    end
  else
    push!(strat_quants, [1])
  end

  for dv_ in dv
    stratified_vpc = VPC_STRAT[]
    obs_vpc_dv = [] 
    for strat in 1:length(strat_quants)

      if stratify_on != nothing
        pop_quantiles = get_simulation_quantiles(sims, reps, dv_, idv_, quantiles, strat_quants[strat],stratify_on[strat])
        quantile_quantiles = get_quant_quantiles(pop_quantiles,reps,idv_,quantiles, strat_quants[strat])
        vpc_strat = get_vpc(quantile_quantiles, data, dv_, idv_, sims, quantiles, strat_quants[strat], stratify_on[strat])
        if data[1].observations != nothing
          obs_quantiles = get_observation_quantiles(data, dv_, idv_, quantiles, strat_quants[strat], stratify_on[strat])
        else
          obs_quantiles = [nothing for i in 1:length(strat_quants[strat])]
        end
      else
        pop_quantiles = get_simulation_quantiles(sims, reps, dv_, idv_, quantiles, strat_quants[strat],nothing)
        quantile_quantiles = get_quant_quantiles(pop_quantiles,reps,idv_,quantiles, strat_quants[strat])
        vpc_strat = get_vpc(quantile_quantiles, data, dv_, idv_, sims, quantiles, strat_quants[strat], nothing)
        if data[1].observations != nothing  
          obs_quantiles = get_observation_quantiles(data, dv_, idv_, quantiles, strat_quants[strat], nothing)
        else
          obs_quantiles = [nothing for i in 1:length(strat_quants[strat])]
        end
      end

      push!(obs_vpc_dv, obs_quantiles)
      push!(stratified_vpc, vpc_strat)
    end
    push!(vpcs, VPC_DV(stratified_vpc, dv_))
    push!(obs_vpc, obs_vpc_dv) 
  end
  VPC(vpcs, OBS_VPC(idv_, obs_vpc), sims, idv, data)
end

# function show(io::IO, mime::MIME"text/plain", vpc::VPC)
#   println(io, "")  
# end

#Recipes for the VPC and subsequent objects that store the quantiles per dv, strata and quantiles
@recipe function f(vpc::VPC; data=vpc.data)
  t = vpc.Obs_vpc.t
  for i in 1:length(vpc.vpc_dv)
    @series begin
      t, vpc.vpc_dv[i], vpc.Obs_vpc.Obs_vpc[i], vpc.idv
    end
  end
  
end

@recipe function f(t, vpc_dv::VPC_DV, data, idv=:time)
  layout --> sum([length(vpc_dv.vpc_strat[i].vpc_quant) for i in 1:length(vpc_dv.vpc_strat)])
  j = 1
  for (i,vpc_strt) in enumerate(vpc_dv.vpc_strat)
    for quant in 1:length(vpc_strt.vpc_quant)
      @series begin
        subplot := j
        if vpc_strt.strat != nothing
          title --> "Stratified on: "*string(vpc_strt.strat, " ", round(vpc_strt.strata_val[quant], sigdigits = 3))
        end
        t, vpc_strt.vpc_quant[quant], data[i][quant], idv
      end
      j += 1
    end
  end
end

@recipe function f(t, vpc_quant::VPC_QUANT, data, idv=:time)
  legend --> false
  lw --> 3
  ribbon := vpc_quant.Fifth_Ninetyfifth
  fillalpha := 0.2
  xlabel --> string(idv)
  ylabel --> "Observations"
  if data != nothing
    for y in [vpc_quant.Fiftieth, data]
      @series begin
        t, y
      end 
    end 
  else 
    t, vpc_quant.Fiftieth
  end
end

@recipe function f(Obs_vpc::OBS_VPC)
  for i in 1:length(Obs_vpc.Obs_vpc)
    for j in 1:length(Obs_vpc.Obs_vpc[i])
      for k in 1:length(Obs_vpc.Obs_vpc[i][j])
        @series begin
          Obs_vpc.t, Obs_vpc.Obs_vpc[i][j][k]
        end
      end 
    end
  end
end

@recipe function f(t, data::OBS_QUANT)
  seriestype := :scatter
  t, data.Observation_Quantiles
end