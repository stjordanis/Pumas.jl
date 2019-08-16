function StatsBase.residuals(fpm::FittedPumasModel)
  # Return the residuals
  return [residuals(fpm.model, subject, fpm.param, vrandeffs) for (subject, vrandeffs) in zip(fpm.data, fpm.vvrandeffs)]
end
function StatsBase.residuals(model::PumasModel, subject::Subject, param::NamedTuple, vrandeffs::AbstractArray)
  rtrf = totransform(model.random(param))
  randeffs = TransformVariables.transform(rtrf, vrandeffs)
  # Calculated the dependent variable distribution objects
  dist = derived_dist(model, subject, param, randeffs)
  # Return the residuals
  return residuals(subject, dist)
end
function StatsBase.residuals(subject::Subject, dist)
  # Return the residuals
  return subject.observations.dv .- mean.(dist.dv)
end
"""
  npde(model, subject, param, randeffs, simulations_count)

To calculate the Normalised Prediction Distribution Errors (NPDE).
"""
function npde(m::PumasModel,
              subject::Subject,
              param::NamedTuple,
              randeffs::NamedTuple,
              nsim::Integer)
  y = subject.observations.dv
  sims = [simobs(m, subject, param, randeffs).observed.dv for i in 1:nsim]
  mean_y = mean(sims)
  cov_y = Symmetric(cov(sims))
  Fcov_y = cholesky(cov_y)
  y_decorr = Fcov_y.U'\(y .- mean_y)

  φ = mean(sims) do y_l
    y_decorr_l = Fcov_y\(y_l .- mean_y)
    Int.(y_decorr_l .< y_decorr)
  end

  return quantile.(Normal(), φ)
end

struct SubjectResidual{T1, T2, T3, T4}
  wres::T1
  iwres::T2
  subject::T3
  approx::T4
end
function wresiduals(fpm::FittedPumasModel, approx=fpm.approx; nsim=nothing)
  subjects = fpm.data
  if approx == fpm.approx
    vvrandeffs = fpm.vvrandeffs
  else
    # re-estimate under approx
    vvrandeffs = [empirical_bayes(fpm.model, subject, fpm.param, approx) for subject in subjects]
  end
  [wresiduals(fpm, subjects[i], vvrandeffs[i], approx; nsim=nsim) for i = 1:length(subjects)]
end
function wresiduals(fpm, subject::Subject, randeffs, approx; nsim=nothing)
  is_sim = nsim == nothing
  if nsim == nothing
    approx = approx
    wres = wresiduals(fpm.model, subject, fpm.param, randeffs, approx)
    iwres = iwresiduals(fpm.model, subject, fpm.param, randeffs, approx)
  else
    approx = nothing
    wres = nothing
    iwres = eiwres(fpm.model, subject, fpm.param, nsim)
  end

  SubjectResidual(wres, iwres, subject, approx)
end

function DataFrames.DataFrame(vresid::Vector{<:SubjectResidual}; include_covariates=true)
  subjects = [resid.subject for resid in vresid]
  df = select!(DataFrame(subjects; include_covariates=include_covariates, include_dvs=false), Not(:evid))

  df[!,:wres] .= vcat((resid.wres for resid in vresid)...)
  df[!,:iwres] .= vcat((resid.iwres for resid in vresid)...)
  df[!,:wres_approx] .= vcat((fill(resid.approx, length(resid.subject.time)) for resid in vresid)...)

  df
end

"""
    restype(approx)

Returns the residual type for the given approximation method.
Can be one of [`FO`](@ref), [`FOCE`](@ref), or [`FOCEI`](@ref).
"""
restype(::FO) = :wres
restype(::FOCE) = :cwres
restype(::FOCEI) = :cwresi

function wresiduals(model, subject, param, randeffs, approx::FO)
  wres(model, subject, param, randeffs)
end
function wresiduals(model, subject, param, randeffs, approx::FOCE)
  cwres(model, subject, param, randeffs)
end
function wresiduals(model, subject, param, randeffs, approx::FOCEI)
  cwresi(model, subject, param, randeffs)
end
function iwresiduals(model, subject, param, randeffs, approx::FO)
  iwres(model, subject, param, randeffs)
end
function iwresiduals(model, subject, param, randeffs, approx::FOCE)
  icwres(model, subject, param, randeffs)
end
function iwresiduals(model, subject, param, randeffs, approx::FOCEI)
  icwresi(model, subject, param, randeffs)
end

"""

  wres(model, subject, param[, rfx])

To calculate the Weighted Residuals (WRES).
"""
function wres(m::PumasModel,
              subject::Subject,
              param::NamedTuple,
              vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FO()))

  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  Ω = Matrix(param.Ω)
  F = ForwardDiff.jacobian(s -> mean.(derived_dist(m, subject, param, (η=s,)).dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return cholesky(V).U'\residuals(subject, dist)
end

"""
  cwres(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals (CWRES).
"""
function cwres(m::PumasModel,
               subject::Subject,
               param::NamedTuple,
               vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCE()))
  dist0 = derived_dist(m, subject, param, (η=zero(vrandeffs),))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  Ω = Matrix(param.Ω)
  F = ForwardDiff.jacobian(s -> mean.(derived_dist(m, subject, param, (η=s,)).dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist0.dv)))
  return cholesky(V).U'\(residuals(subject, dist) .+ F*vrandeffs)
end

"""
  cwresi(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals with Interaction (CWRESI).
"""

function cwresi(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCEI()))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  Ω = Matrix(param.Ω)
  F = ForwardDiff.jacobian(s -> mean.(derived_dist(m, subject, param, (η=s,)).dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return cholesky(V).U'\(residuals(subject, dist) .+ F*vrandeffs)
end

"""
  pred(model, subject, param[, rfx])

To calculate the Population Predictions (PRED).
"""
function pred(m::PumasModel,
              subject::Subject,
              param::NamedTuple,
              vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FO()))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  return mean.(dist.dv)
end


"""
  cpred(model, subject, param[, rfx])

To calculate the Conditional Population Predictions (CPRED).
"""
function cpred(m::PumasModel,
               subject::Subject,
               param::NamedTuple,
               vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCE()))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  F = ForwardDiff.jacobian(s -> mean.(derived_dist(m, subject, param, (η=s,)).dv), vrandeffs)
  return mean.(dist.dv) .- F*vrandeffs
end

"""
  cpredi(model, subject, param[, rfx])

To calculate the Conditional Population Predictions with Interaction (CPREDI).
"""
function cpredi(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCEI()))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  F = ForwardDiff.jacobian(s -> mean.(derived_dist(m, subject, param, (η=s,)).dv), vrandeffs)
  return mean.(dist.dv) .- F*vrandeffs
end

"""
  epred(model, subject, param[, rfx], simulations_count)

To calculate the Expected Simulation based Population Predictions.
"""
function epred(m::PumasModel,
               subject::Subject,
               param::NamedTuple,
               randeffs::NamedTuple,
               nsim::Integer)
  sims = [simobs(m, subject, param, randeffs).observed.dv for i in 1:nsim]
  return mean(sims)
end

"""
  iwres(model, subject, param[, rfx])

To calculate the Individual Weighted Residuals (IWRES).
"""
function iwres(m::PumasModel,
               subject::Subject,
               param::NamedTuple,
               vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FO()))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  return residuals(subject, dist) ./ std.(dist.dv)
end

"""
  icwres(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals (ICWRES).
"""
function icwres(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCE()))
  dist0 = derived_dist(m, subject, param, (η=zero(vrandeffs),))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  return residuals(subject, dist) ./ std.(dist0.dv)
end

"""
  icwresi(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals with Interaction (ICWRESI).
"""
function icwresi(m::PumasModel,
                 subject::Subject,
                 param::NamedTuple,
                 vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCEI()))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  return residuals(subject, dist) ./ std.(dist.dv)
end

"""
  eiwres(model, subject, param[, rfx], simulations_count)

To calculate the Expected Simulation based Individual Weighted Residuals (EIWRES).
"""
function eiwres(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                nsim::Integer)
  yi = subject.observations.dv
  dist = derived_dist(m, subject, param, sample_randeffs(m, param))
  sims_sum = (yi .- mean.(dist.dv))./std.(dist.dv)
  for i in 2:nsim
    dist = derived_dist(m, subject, param, sample_randeffs(m, param))
    sims_sum .+= (yi .- mean.(dist.dv))./std.(dist.dv)
  end
  return sims_sum ./ nsim
end

function ipred(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FO()))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  return mean.(dist.dv)
end

function cipred(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCE()))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  return mean.(dist.dv)
end

function cipredi(m::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCEI()))
  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  return mean.(dist.dv)
end

function ηshrinkage(m::PumasModel,
                    data::Population,
                    param::NamedTuple,
                    approx::LikelihoodApproximation)
  sd_randeffs = std([empirical_bayes(m, subject, param, approx) for subject in data])
  Ω = Matrix(param.Ω)
  return  1 .- sd_randeffs ./ sqrt.(diag(Ω))
end

function ϵshrinkage(m::PumasModel,
                    data::Population,
                    param::NamedTuple,
                    approx::FOCEI,
                    randeffs=[empirical_bayes(m, subject, param, FOCEI()) for subject in data])
  1 - std(vec(VectorOfArray([icwresi(m, subject, param, vrandeffs) for (subject, vrandeffs) in zip(data, randeffs)])), corrected = false)
end

function ϵshrinkage(m::PumasModel,
                    data::Population,
                    param::NamedTuple,
                    approx::FOCE,
                    randeffs=[empirical_bayes(m, subject, param, FOCE()) for subject in data])
  1 - std(vec(VectorOfArray([icwres(m, subject, param, vrandeffs) for (subject,vrandeffs) in zip(data, randeffs)])), corrected = false)
end

function StatsBase.aic(m::PumasModel,
                       data::Population,
                       param::NamedTuple,
                       approx::LikelihoodApproximation)
  numparam = TransformVariables.dimension(totransform(m.param))
  2*(marginal_nll(m, data, param, approx) + numparam)
end

function StatsBase.bic(m::PumasModel,
                       data::Population,
                       param::NamedTuple,
                       approx::LikelihoodApproximation)
  numparam = TransformVariables.dimension(totransform(m.param))
  2*marginal_nll(m, data, param, approx) + numparam*log(sum(t -> length(t.time), data))
end

### Predictions
struct SubjectPrediction{T1, T2, T3, T4}
  pred::T1
  ipred::T2
  subject::T3
  approx::T4
end

function StatsBase.predict(model::PumasModel, subject::Subject, param, approx, vrandeffs=empirical_bayes(model, subject, param, approx))
  pred = _predict(model, subject, param, approx, vrandeffs)
  ipred = _ipredict(model, subject, param, approx, vrandeffs)
  SubjectPrediction(pred, ipred, subject, approx)
end

function StatsBase.predict(fpm::FittedPumasModel, approx=fpm.approx; nsim=nothing, timegrid=false, newdata=false, useEBEs=true)
  if !useEBEs
    error("Sampling from the omega distribution is not yet implemented.")
  end
  if !(newdata==false)
    error("Using data different than that used to fit the model is not yet implemented.")
  end
  if !(timegrid==false)
    error("Using custom time grids is not yet implemented.")
  end

  subjects = fpm.data

  if approx == fpm.approx
    vvrandeffs = fpm.vvrandeffs
  else
    # re-estimate under approx
    vvrandeffs = [empirical_bayes(fpm.model, subject, fpm.param, approx) for subject in subjects]
  end
  [predict(fpm.model, subjects[i], fpm.param, approx, vvrandeffs[i]) for i = 1:length(subjects)]
end

function _predict(model, subject, param, approx::FO, vrandeffs)
  pred(model, subject, param, vrandeffs)
end
function _ipredict(model, subject, param, approx::FO, vrandeffs)
  ipred(model, subject, param, vrandeffs)
end

function _predict(model, subject, param, approx::Union{FOCE, Laplace}, vrandeffs)
  cpred(model, subject, param, vrandeffs)
end
function _ipredict(model, subject, param, approx::Union{FOCE, Laplace}, vrandeffs)
  cipred(model, subject, param, vrandeffs)
end

function _predict(model, subject, param, approx::Union{FOCEI, LaplaceI}, vrandeffs)
  cpredi(model, subject, param, vrandeffs)
end
function _ipredict(model, subject, param, approx::Union{FOCEI, LaplaceI}, vrandeffs)
  cipredi(model, subject, param, vrandeffs)
end

function epredict(fpm, subject, vrandeffs, nsim::Integer)
  epred(fpm.model, subjects, fpm.param, (η=vrandeffs,), nsim)
end

function DataFrames.DataFrame(vpred::Vector{<:SubjectPrediction}; include_covariates=true)
  subjects = [pred.subject for pred in vpred]
  df = select!(DataFrame(subjects; include_covariates=include_covariates, include_dvs=false), Not(:evid))

  df[!,:pred] .= vcat((pred.pred for pred in vpred)...)
  df[!,:ipred] .= vcat((pred.ipred for pred in vpred)...)
  df[!,:pred_approx] .= vcat((fill(pred.approx, length(pred.subject.time)) for pred in vpred)...)

  df
end

struct SubjectEBES{T1, T2, T3}
  ebes::T1
  subject::T2
  approx::T3
end
function empirical_bayes(fpm::FittedPumasModel, approx=fpm.approx)
  subjects = fpm.data

  if approx == fpm.approx
    ebes = fpm.vvrandeffs
    return [SubjectEBES(e, s, approx) for (e, s) in zip(ebes, subjects)]
  else
    # re-estimate under approx
    return [SubjectEBES(empirical_bayes(fpm.model, subject, fpm.param, approx), subject, approx) for subject in subjects]
  end
end

function DataFrames.DataFrame(vebes::Vector{<:SubjectEBES}; include_covariates=true)
  subjects = [ebes.subject for ebes in vebes]
  df = select!(DataFrame(subjects; include_covariates=include_covariates, include_dvs=false), Not(:evid))
  for i = 1:length(first(vebes).ebes)
    df[!,Symbol("ebe_$i")] .= vcat((fill(ebes.ebes[i], length(ebes.subject.time)) for ebes in vebes)...)
  end
  df[!,:ebes_approx] .= vcat((fill(ebes.approx, length(ebes.subject.time)) for ebes in vebes)...)

  df
end

struct FittedPumasModelInspection{T1, T2, T3, T4}
  o::T1
  pred::T2
  wres::T3
  ebes::T4
end
StatsBase.predict(i::FittedPumasModelInspection) = i.pred
wresiduals(i::FittedPumasModelInspection) = i.wres
empirical_bayes(i::FittedPumasModelInspection) = i.ebes

function inspect(fpm; pred_approx=fpm.approx, infer_approx=fpm.approx,
                    wres_approx=fpm.approx, ebes_approx=fpm.approx)
  print("Calculating: ")
  print("predictions")
  pred = predict(fpm, pred_approx)
  print(", weighted residuals")
  res = wresiduals(fpm, wres_approx)
  print(", empirical bayes")
  ebes = empirical_bayes(fpm, ebes_approx)
  println(". Done.")
  FittedPumasModelInspection(fpm, pred, res, ebes)
end
function DataFrames.DataFrame(i::FittedPumasModelInspection; include_covariates=true)
  pred_df = DataFrame(i.pred; include_covariates=include_covariates)
  res_df = select!(select!(DataFrame(i.wres; include_covariates=false), Not(:id)), Not(:time))
  ebes_df = select!(select!(DataFrame(i.ebes; include_covariates=false), Not(:id)), Not(:time))

  df = hcat(pred_df, res_df, ebes_df)
end


################################################################################
#                              Plotting functions                              #
################################################################################

########################################
#   Convergence plot infrastructure    #
########################################

"""
    _objectivefunctionvalues(obj)

Returns the objective function values during optimization.
Must return a `Vector{Number}`.
"""
_objectivefunctionvalues(f::FittedPumasModel) = getproperty.(f.optim.trace, :value)

"""
    _convergencedata(obj; metakey="x")

Returns the "timeseries" of optimization as a matrix, with series as columns.
!!! warn
    This must return parameter data in the same order that [`_paramnames`](@ref)
    returns names.
"""
function _convergencedata(f::FittedPumasModel; metakey="x")

  metakey != "x" && return transpose(hcat(getindex.(getproperty.(f.optim.trace, :metadata), metakey))...)

  trf  = totransform(f.model.param)         # get the transform which has been applied to the params
  itrf = toidentitytransform(f.model.param) # invert the param transform

  return transpose(                                     # return series as columns
              hcat(TransformVariables.inverse.(         # apply the inverse of the given transform to the data.
                  Ref(itrf),                            # wrap in a `Ref`, to avoid broadcasting issues
                  TransformVariables.transform.(        # apply the initial transform to the process
                      Ref(trf),                         # again - make sure no broadcasting across the `TransformTuple`
                      getindex.(                        # get every `x` vector from the metadata of the trace
                          getproperty.(                 # get the metadata of each trace element
                              f.optim.trace, :metadata  # getproperty expects a `Symbol`
                              ),
                          metakey                           # property x is a key for a `Dict` - hence getindex
                          )
                      )
                  )...                                  # splat to get a matrix out
              )
          )
end

"""
    _paramnames(obj)

Returns the names of the parameters which convergence is being checked for.
!!! warn
    This must return parameter names in the same order that [`_convergencedata`](@ref)
    returns data.
"""
function _paramnames(f::FittedPumasModel)
  paramnames = [] # empty array, will fill later
  for (paramname, paramval) in pairs(f.param) # iterate through the parameters
    # decompose all parameters (matrices, etc.) into scalars and name them appropriately
    _push_varinfo!(paramnames, [], nothing, nothing, paramname, paramval, nothing, nothing)
  end
  return paramnames
end

########################################
#             Type recipe              #
########################################

@recipe function f(arg::FittedPumasModel; params = Union{Symbol, String}[])

    names = _paramnames(arg)      # a wrapper function around some logic to make use of multiple dispatch

    str_params = string.(params)

    data  = _convergencedata(arg) # again, a wrapper function.

    inds = []

    !isempty(params) && (inds = findall(x -> !(x in str_params), names))

    finalnames = deleteat!(names, inds)

    finaldata = @view(data[(x -> !(x in inds)).(axes(data)[1]), :])

    layout --> good_layout(length(finalnames) + 1)  # good_layout simply overrides the Plots layouter for n <= 4

    legend := :none # clutters up the plot too much

    # also plot the observed function value from Optim
    @series begin
        title := "Objective function"
        subplot := length(names) + 1
        link := :x # link the x-axes, there should be no difference

        (_objectivefunctionvalues(arg)) # the observed values
    end

    # Iterate through three variables simultaneously:
    # - the index of the data (to assign to the subplot),
    # - the name of the parameter (as determined by `_paramnames`)
    # - the convergence data to plot as series vectors
    for (i, name, trace) in zip(eachindex(names), names, eachslice(data; dims = 2))
        @series begin
            title := name
            subplot := i
            link := :x
            # @show name i trace # debug here
            trace # return series data to be plotted.  We let the user specify the seriestype.
        end
    end

    primary := false # do not add a legend entry here, do other nice things that Plots does as well
end
