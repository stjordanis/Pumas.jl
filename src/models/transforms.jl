using TransformVariables

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
  inverse_eltype(first(t.a), first(y))
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


function TransformVariables.inverse_eltype(t::ConstantTransform, y::AbstractArray)
  Float64
end
function TransformVariables.inverse(t::ConstantTransform, v)
  @assert t.val == v
  Float64[]
end
function TransformVariables.inverse!(x::AbstractArray,t::ConstantTransform, v)
  @assert t.val == v
  x
end

struct PSDCholeskyFactor <: TransformVariables.VectorTransform
    n::Int
end

TransformVariables.dimension(t::PSDCholeskyFactor) = (t.n*(t.n+1))>>1

function TransformVariables.transform_with(flag::TransformVariables.LogJacFlag, t::PSDCholeskyFactor,
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
            ℓ += ℓi
        end
        index += 1
    end
    Cholesky(U, 'U', 0), ℓ
end

# a bit of type piracy
# we need to use a slightly different density here,
# since it is wrt the Cholesky, not the 
function Distributions.logpdf(d::Wishart, C::Cholesky)
    df = d.df
    p = dim(d)
    sum((df-i)*log(C.UL[i,i]) for i = 1:p) - 0.5 * tr(d.S \ Matrix(C)) - d.c0
end
function Distributions.logpdf(d::InverseWishart, C::Cholesky)
    df = d.df
    p = dim(d)
    sum((-df-i)*log(C.UL[i,i]) for i = 1:p) - 0.5 * tr(C \ Matrix(d.Ψ)) - d.c0
end


totransform(p::ParamSet) = as(map(totransform, p.params))
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

totransform(d::PSDDomain) = PSDCholeskyFactor(size(d.init,1))

totransform(d::Distribution) = totransform(Domain(d))
totransform(d::MvNormal) = as(Array, length(d))
totransform(c::Constrained) = totransform(c.domain)
