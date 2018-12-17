using LogDensityProblems
import LogDensityProblems: AbstractLogDensityProblem, dimension, logdensity

struct BayesModel{M,D} <: AbstractLogDensityProblem
  model::M
  data::D
end

function dimension(b::BayesModel)
  param = b.model.param
  t_param = totransform(param)
  m = dimension(t_param)
  x = TransformVariables.transform(t_param, zeros(m))
  rfx = b.model.random(x)
  t_rfx = totransform(rfx)
  n = dimension(t_rfx)

  return m + n*length(b.data.subjects)
end

function logdensity(::Type{LogDensityProblems.Value}, b::BayesModel, x::AbstractVector)
  v = logdensity(b, x)
  v < Inf || @show x
  LogDensityProblems.Value(v)
end

function logdensity(b::BayesModel, v::AbstractVector)
  param = b.model.param
  t_param = totransform(param)
  m = dimension(t_param)
  x, j_x = TransformVariables.transform_and_logjac(t_param, @view v[1:m])
  ℓ_param = _lpdf(param.params, x) + j_x

  rfx = b.model.random(x)
  t_rfx = totransform(rfx)
  n = dimension(t_rfx)
  ℓ_rfx = sum(enumerate(b.data.subjects)) do (i,subject)
    y, j_y = TransformVariables.transform_and_logjac(t_rfx, @view v[m+(i-1)*n+1:m+i*n])
    j_y - PuMaS.penalized_conditional_nll(b.model, subject, x, y)
  end
  ℓ_param + ℓ_rfx
end
