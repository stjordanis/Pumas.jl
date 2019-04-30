using LinearAlgebra
export ParamSet, ConstDomain, RealDomain, VectorDomain, PSDDomain, PDiagDomain, Constrained

abstract type Domain end

Domain(d::Domain) = d

"""
    @param x = val
    @param x ∈ ConstDomain(val)

Specifies a parameter as a constant.
"""
struct ConstDomain{T} <: Domain
  val::T
end
init(d::ConstDomain) = d.val

"""
    @param x ∈ RealDomain(;lower=-Inf,upper=+Inf,init=0)

Specifies a parameter as a real value. `lower` and `upper` are the respective bounds, `init` is the value used as the initial guess in the optimisation.
"""
struct RealDomain{T} <: Domain
  lower::T
  upper::T
  init::T
end
RealDomain(;lower=-Inf,upper=+Inf,init=0.) = RealDomain(promote(lower, upper, init)...)
init(d::RealDomain) = d.init


"""
    @param x ∈ VectorDomain(n::Int; lower=-Inf,upper=+Inf,init=0)

Specifies a parameter as a real vector of length `n`. `lower` and `upper` are the respective bounds, `init` is the value used as the initial guess in the optimisation.
"""
struct VectorDomain{L,T} <: Domain
  lower::L
  upper::L
  init::T
end

_vec(n, x::AbstractVector) = x
_vec(n, x) = fill(x, n)

VectorDomain(n::Int; lower=-Inf,upper=+Inf,init=0.0) = VectorDomain(promote(_vec(n,lower), _vec(n,upper))..., _vec(n,init))

init(d::VectorDomain) = d.init


"""
    @param x ∈ PSDDomain(n::Int; init=Matrix{Float64}(I, n, n))

Specifies a parameter as a symmetric `n`-by-`n` positive semidefinite matrix.
"""
struct PSDDomain{T} <: Domain
  init::T
end
PSDDomain(; init=nothing) = PSDDomain(init)
PSDDomain(n::Int)         = PSDDomain(init=Matrix{Float64}(I, n, n))

init(d::PSDDomain) = d.init


"""
    @param x ∈ PDiagDomain(n::Int; init=ones(n))

Specifies a parameter as a positive diagonal matrix, with diagonal elements
specified by `init`.
"""
struct PDiagDomain{T} <: Domain
  init::T
end
PDiagDomain(; init=missing) = PDiagDomain(PDMats.PDiagMat(init))
PDiagDomain(n::Int)         = PDiagDomain(init=ones(n))

init(d::PDiagDomain) = d.init


# domains of random variables
function Domain(d::MvNormal)
  n = length(d)
  VectorDomain(fill(-Inf, n), fill(Inf, n), mean(d))
end
Domain(d::InverseWishart) = PSDDomain(Distributions.dim(d))

function Domain(d::ContinuousUnivariateDistribution)
  RealDomain(minimum(d), maximum(d), median(d))
end


struct ParamSet{T}
  params::T
end

domains(p::ParamSet) = map(Domain, p.params)

init(p::ParamSet) = map(init, domains(p))

Base.rand(p::ParamSet) = map(rand, p.params)


struct Constrained{D<:Distribution,M<:Domain}
  dist::D
  domain::M
end

Constrained(dist::MvNormal; lower=-Inf, upper=Inf, init=0.0) =
  Constrained(dist, VectorDomain(length(dist); lower=lower, upper=upper, init=init))

Constrained(dist::ContinuousUnivariateDistribution; lower=-Inf, upper=Inf, init=0.0) =
  Constrained(dist, RealDomain(; lower=lower, upper=upper, init=init))

Domain(c::Constrained) = c.domain

# obviously wrong, but fine as long as parameters are constant
# need to enforce this somehow
Distributions.logpdf(d::Constrained, x) = logpdf(d.dist, x)
