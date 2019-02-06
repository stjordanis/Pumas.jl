using LogDensityProblems
import LogDensityProblems: AbstractLogDensityProblem, dimension, logdensity


function dim_param(m::PKPDModel)
  t_param = totransform(m.param)
  dimension(t_param)
end

function dim_rfx(m::PKPDModel)
  x = init_param(m)
  rfx = m.random(x)
  t_rfx = totransform(rfx)
  dimension(t_rfx)
end


struct BayesModel{M,D,B,C,R} <: LogDensityProblems.AbstractLogDensityProblem
  model::M
  data::D
  dim_param::Int
  dim_rfx::Int
  buffer::B
  cfg::C
  res::R
end

function BayesModel(model, data)
  m = dim_param(model)
  n = dim_rfx(model)
  buffer = zeros(m + n)
  cfg = ForwardDiff.GradientConfig(nothing, buffer)
  res = DiffResults.GradientResult(buffer)
  BayesModel(model, data, m, n, buffer, cfg, res)
end

function LogDensityProblems.dimension(b::BayesModel)
  b.dim_param + b.dim_rfx * length(b.data.subjects)
end

function LogDensityProblems.logdensity(::Type{LogDensityProblems.Value}, b::BayesModel, v::AbstractVector)
  try
    m = b.dim_param
    param = b.model.param
    t_param = totransform(param)
    x, j_x = TransformVariables.transform_and_logjac(t_param, @view v[1:m])
    ℓ_param = _lpdf(param.params, x) + j_x

    n = b.dim_rfx
    rfx = b.model.random(x)
    t_rfx = totransform(rfx)
    n = dimension(t_rfx)
    ℓ_rfx = sum(enumerate(b.data.subjects)) do (i,subject)
      y, j_y = TransformVariables.transform_and_logjac(t_rfx, @view v[(m+(i-1)*n) .+ (1:n)])
      j_y - PuMaS.penalized_conditional_nll(b.model, subject, x, y)
    end
    ℓ = ℓ_param + ℓ_rfx
    return LogDensityProblems.Value(isnan(ℓ) ? -Inf : ℓ)
  catch e
    if e isa ArgumentError
      return LogDensityProblems.Value(-Inf)
    else
      rethrow(e)
    end
  end
end

function LogDensityProblems.logdensity(::Type{LogDensityProblems.ValueGradient}, b::BayesModel, v::AbstractVector)
  ∇ℓ = zeros(size(v))
  try

    param = b.model.param
    t_param = totransform(param)
    m = b.dim_param
    n = b.dim_rfx

    function L_param(u)
      x, j_x = TransformVariables.transform_and_logjac(t_param, u)
      _lpdf(param.params, x) + j_x
    end
    fill!(b.buffer, 0.0)
    copyto!(b.buffer, 1, v, 1, m)

    ForwardDiff.gradient!(b.res, L_param, b.buffer, b.cfg)

    ℓ = DiffResults.value(b.res)
    ∇ℓ[1:m] .= @view DiffResults.gradient(b.res)[1:m]

    for (i, subject) in enumerate(b.data.subjects)
      function L_rfx(u)
        x = TransformVariables.transform(t_param, @view u[1:m])
        rfx = b.model.random(x)
        t_rfx = totransform(rfx)
        y, j_y = TransformVariables.transform_and_logjac(t_rfx, @view u[m .+ (1:n)])
        j_y - PuMaS.penalized_conditional_nll(b.model, subject, x, y)
      end
      copyto!(b.buffer, m+1, v, m+(i-1)*n+1, n)

      ForwardDiff.gradient!(b.res, L_rfx, b.buffer, b.cfg)

      ℓ += DiffResults.value(b.res)
      ∇ℓ[1:m] .+= @view DiffResults.gradient(b.res)[1:m]
      copyto!(∇ℓ, m+(i-1)*n+1, DiffResults.gradient(b.res), m+1, n)
    end
    return LogDensityProblems.ValueGradient(isnan(ℓ) ? -Inf : ℓ, ∇ℓ)
  catch e
    if e isa ArgumentError
      return LogDensityProblems.ValueGradient(-Inf, ∇ℓ)
    else
      rethrow(e)
    end
  end
end
