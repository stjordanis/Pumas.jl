import TransformVariables
using TransformVariables: as, asℝ₊, ∞

struct ElementArrayTransform{T<:TransformVariables.AbstractTransform,N} <: TransformVariables.VectorTransform
  a::Array{T,N}
end

TransformVariables.dimension(t::ElementArrayTransform) = length(t.a)

TransformVariables.as(x::Array{<:TransformVariables.AbstractTransform}) = ElementArrayTransform(x)

function TransformVariables.transform_with(flag::TransformVariables.LogJacFlag, t::ElementArrayTransform, x::TransformVariables.RealVector)
  # currently support only scalars
  dims = size(t.a)
  I = reshape(range(firstindex(x); length = prod(dims), step = 1), dims)
  yℓ = map((i,ti) -> TransformVariables.transform_with(flag, ti, TransformVariables.view_into(x, i, 1)), I, t.a)
  first.(yℓ), sum(last, yℓ)
end

function TransformVariables.inverse_eltype(t::ElementArrayTransform, y::AbstractArray)
  TransformVariables.inverse_eltype(first(t.a), first(y))
end
function TransformVariables.inverse!(x::AbstractArray, t::ElementArrayTransform, y::AbstractArray)
  map!(TransformVariables.inverse, x, t.a, y)
end
function TransformVariables.inverse(t::ElementArrayTransform, y::AbstractArray)
  map(TransformVariables.inverse, t.a, y)
end

struct ConstantTransform{T} <: TransformVariables.AbstractTransform
  val::T
end
TransformVariables.dimension(t::ConstantTransform) = 0

function TransformVariables.transform_with(flag::TransformVariables.NoLogJac, t::ConstantTransform, x::TransformVariables.RealVector)
  t.val, flag
end

function TransformVariables.transform_with(flag::TransformVariables.LogJac, t::ConstantTransform, x::TransformVariables.RealVector)
  t.val, zero(eltype(x))
end


# function TransformVariables.inverse_eltype(t::ConstantTransform, y::T) where T<:Real
  # T
# end
TransformVariables.inverse_eltype(t::ConstantTransform, ::Any) = eltype(t.val)
function TransformVariables.inverse(t::ConstantTransform, v)
  @assert t.val == v
  Float64[]
end
function TransformVariables.inverse!(x::AbstractArray,t::ConstantTransform, v)
  @assert t.val == v
  x
end



struct PSDTransform <: TransformVariables.VectorTransform
    n::Int
end
TransformVariables.dimension(t::PSDTransform) = (t.n*(t.n+1))>>1

function TransformVariables.transform_with(flag::TransformVariables.LogJacFlag, t::PSDTransform,
                                           x::TransformVariables.RealVector{T}) where T
    n = t.n
    ℓ = TransformVariables.logjac_zero(flag, T)
    U = zeros(typeof(√one(T)), n, n)
    index = TransformVariables.firstindex(x)
    @inbounds for col in 1:n
        r = one(T)
        for row in 1:(col-1)
            U[row, col] = x[index]
            index += 1
        end
        if flag isa TransformVariables.NoLogJac
            U[col, col] = TransformVariables.transform(asℝ₊, x[index])
        else
            U[col, col], ℓi = TransformVariables.transform_and_logjac(asℝ₊, x[index])
            ℓ += (n-col+2)*ℓi
        end
        index += 1
    end
    PDMat(Cholesky(U, 'U', 0)), ℓ
end

TransformVariables.inverse_eltype(::PSDTransform, y::PDMat{T}) where T = T
TransformVariables.inverse_eltype(::PSDTransform, y::AbstractMatrix{T}) where T = T

function TransformVariables.inverse!(x::AbstractVector, t::PSDTransform, y::PDMat)
  index = TransformVariables.firstindex(x)
  F = y.chol
  @assert F.uplo === 'U'
  U = F.U
  @inbounds for col in 1:t.n
    for row in 1:(col-1)
      x[index] = U[row, col]
      index += 1
    end
    x[index] = TransformVariables.inverse(asℝ₊, U[col,col])
    index += 1
  end
  return x
end
TransformVariables.inverse!(x::AbstractVector, t::PSDTransform, y::Matrix) = TransformVariables.inverse!(x, t, PDMat(y))



struct VechTransform <: TransformVariables.VectorTransform
    n::Int
end
TransformVariables.dimension(t::VechTransform) = (t.n*(t.n+1))>>1

function TransformVariables.transform_with(flag::TransformVariables.LogJacFlag, t::VechTransform,
                                           x::TransformVariables.RealVector{T}) where T
  n = t.n
  ℓ = TransformVariables.logjac_zero(flag, T)
  M = zeros(typeof(√one(T)), n, n)
  index = TransformVariables.firstindex(x)
  @inbounds for col in 1:n
    for row in col:n
      M[row, col] = x[index]
      index += 1
    end
    if !(flag isa TransformVariables.NoLogJac)
      error("not support")
    end
  end
  (LinearAlgebra.copytri!(M, 'L')), ℓ
end

TransformVariables.inverse_eltype(::VechTransform, y::PDMat{T}) where T = T
TransformVariables.inverse_eltype(::VechTransform, y::AbstractMatrix{T}) where T = T

function TransformVariables.inverse!(x::AbstractVector, t::VechTransform, y::PDMat)
  index = TransformVariables.firstindex(x)
  M = y.mat
  @inbounds for col in 1:t.n
    for row in col:t.n
      x[index] = M[row, col]
      index += 1
    end
  end
  return x
end
TransformVariables.inverse!(x::AbstractVector, t::VechTransform, y::Matrix) = TransformVariables.inverse!(x, t, PDMat(y))



struct PDiagTransform  <: TransformVariables.VectorTransform
  n::Int
end

TransformVariables.dimension(t::PDiagTransform) = t.n

function TransformVariables.transform_with(flag::TransformVariables.LogJacFlag, t::PDiagTransform,
                                           x::TransformVariables.RealVector{T}) where T
    n = t.n
    ℓ = TransformVariables.logjac_zero(flag, T)
    d = zeros(typeof(√one(T)), n)
    index = TransformVariables.firstindex(x)
    @inbounds for col in 1:n
        if flag isa TransformVariables.NoLogJac
            d[col] = TransformVariables.transform(asℝ₊, x[index])
        else
            d[col], ℓi = TransformVariables.transform_and_logjac(asℝ₊, x[index])
            ℓ += ℓi
        end
        index += 1
    end
    PDiagMat(d), ℓ
end

TransformVariables.inverse_eltype(::PuMaS.PDiagTransform, y::PDiagMat{T}) where T = T
TransformVariables.inverse_eltype(::PuMaS.PDiagTransform, y::Diagonal{T}) where T = T

function TransformVariables.inverse!(x::AbstractVector, t::PDiagTransform, y::PDiagMat)
  index = TransformVariables.firstindex(x)
  d = y.diag
  # FIXME! Add @inbounds when we know it works
  for col in 1:t.n
    x[index] = TransformVariables.inverse(asℝ₊, d[col])
    index += 1
  end
  return x
end
TransformVariables.inverse!(x::AbstractVector, t::PDiagTransform, y::Diagonal) =
  TransformVariables.inverse!(x, t, PDiagMat(y.diag))


struct DiagonalTransform  <: TransformVariables.VectorTransform
  n::Int
end

TransformVariables.dimension(t::DiagonalTransform) = t.n

function TransformVariables.transform_with(flag::TransformVariables.LogJacFlag, t::DiagonalTransform,
                                           x::TransformVariables.RealVector{T}) where T
    n = t.n
    ℓ = TransformVariables.logjac_zero(flag, T)
    d = zeros(typeof(√one(T)), n)
    index = TransformVariables.firstindex(x)
    @inbounds for col in 1:n
        if flag isa TransformVariables.NoLogJac
            d[col] = x[index]
        else
            error("not supported")
        end
        index += 1
    end
    PDiagMat(d), ℓ
end

TransformVariables.inverse_eltype(::PuMaS.DiagonalTransform, y::Union{Diagonal{T},PDiagMat{T}}) where T = T

function TransformVariables.inverse!(x::AbstractVector, t::DiagonalTransform, y::PDiagMat)
  index = TransformVariables.firstindex(x)
  d = y.diag
  # FIXME! Add @inbounds when we know it works
  for col in 1:t.n
    x[index] = d[col]
    index += 1
  end
  return x
end
TransformVariables.inverse!(x::AbstractVector, t::DiagonalTransform, y::Diagonal) = TransformVariables.inverse!(x, t, PDiagMat(y.diag))


function Distributions.logpdf(d::InverseWishart, M::PDMat)
    p = dim(d)
    df = d.df
    Xcf = M.chol
    # we use the fact: tr(Ψ * inv(X)) = tr(inv(X) * Ψ) = tr(X \ Ψ)
    Ψ = Matrix(d.Ψ)
    -0.5 * ((df + p + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) - d.c0
end

totransform(d::ConstDomain) = ConstantTransform(d.val)
function totransform(d::RealDomain)
  if d.lower == -Inf
    if d.upper == Inf
      as(Real,-∞,∞)
    else
      as(Real,-∞,d.upper)
    end
  else
    if d.upper == Inf
      as(Real,d.lower,∞)
    else
      as(Real,d.lower,d.upper)
    end
  end
end

function totransform(d::VectorDomain)
  as(map((lo,hi) -> totransform(RealDomain(lower=lo,upper=hi)), d.lower, d.upper))
end


totransform(d::PSDDomain) = PSDTransform(size(d.init,1))
totransform(d::PDiagDomain) = PDiagTransform(length(d.init.diag))

totransform(d::Distribution) = totransform(Domain(d))
totransform(d::MvNormal) = as(Array, length(d))
totransform(c::Constrained) = totransform(c.domain)

totransform(p::ParamSet) = as(map(totransform, p.params))

# When computing the Hessian of the population parameters, we'd like to convert
# fixeffs back and forth between a NamedTyple and a Vector in order to utilize
# numerical or automatic differentiation of a all population parameters. For that
# we need some convenience functions
toidentitytransform(p::ParamSet) = as(map(toidentitytransform, p.params))
toidentitytransform(p::ConstDomain) = totransform(p)
toidentitytransform(::RealDomain) = as(Real,-∞,∞)
toidentitytransform(d::VectorDomain) = as(Vector, length(d.lower))
toidentitytransform(d::PSDDomain) = VechTransform(size(d.init, 1))
toidentitytransform(d::PDiagDomain) = DiagonalTransform(size(d.init, 1))

# When using TransformVariables for transforming standard errors, we need to
# ensure that ConstDomains return NaNs.
tostderrortransform(p::ParamSet)    = as(map(tostderrortransform, p.params))
tostderrortransform(d::Domain)      = toidentitytransform(d)
tostderrortransform(d::ConstDomain) = ConstantTransform(d.val*NaN32)
