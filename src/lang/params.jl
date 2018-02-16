export ParamSet, ConstDomain, RealDomain, VectorDomain, PSDDomain

abstract type Domain end

"""
    @param x = val
    @param x ∈ ConstDomain(val)

Specifies a parameter as a constant.
"""
struct ConstDomain{T} <: Domain
    val::T
end
init(d::ConstDomain) = d.val
packlen(d::ConstDomain) = 0

pack_lower!(v, d::ConstDomain) = nothing
pack_upper!(v, d::ConstDomain) = nothing
pack!(v, d::ConstDomain, x) = nothing
unpack(v, d::ConstDomain) = d.val

"""
    @param x ∈ RealDomain(;lower=-Inf,upper=+Inf,init=0)

Specifies a parameter as a real value. `lower` and `upper` are the respective bounds, `init` is the value used as the initial guess in the optimisation.
"""
struct RealDomain{T} <: Domain
    lower::T
    upper::T
    init::T
end
RealDomain(;lower=-Inf,upper=+Inf,init=0) = RealDomain(lower, upper, init)
init(d::RealDomain) = d.init
packlen(d::RealDomain) = 1

pack_lower!(v, d::RealDomain) = v[1] = d.lower
pack_upper!(v, d::RealDomain) = v[1] = d.upper
pack!(v, d::RealDomain, x) = v[1] = x
unpack(v, d::RealDomain) = v[1]


"""
    @param x ∈ Vector(n::Int; lower=-Inf,upper=+Inf,init=0)

Specifies a parameter as a real vector of length `n`. `lower` and `upper` are the respective bounds, `init` is the value used as the initial guess in the optimisation.
"""
struct VectorDomain{L,T} <: Domain
    lower::L
    upper::L
    init::T
end

_vec(n, x::AbstractVector) = x
_vec(n, x) = fill(x, n)

VectorDomain(n::Int; lower=-Inf,upper=+Inf,init=0.0) = VectorDomain(_vec(n,lower), _vec(n,upper), _vec(n,init))

init(d::VectorDomain) = d.init
packlen(d::VectorDomain) = length(d.init)

pack_lower!(v, d::VectorDomain) = v[:] = d.lower
pack_upper!(v, d::VectorDomain) = v[:] = d.upper
pack!(v, d::VectorDomain, x) = v[:] = x
unpack(v, d::VectorDomain) = copy(v)


"""
    @param x ∈ PSDDomain(n::Int; init=eye(n))

Specifies a parameter as a symmetric `n`-by-`n` positive semidefinite matrix.
"""
struct PSDDomain{T} <: Domain
    init::T
end
PSDDomain(n::Int; init=eye(n)) = PSDDomain(PDMat(init))


init(d::PSDDomain) = d.init
function packlen(d::PSDDomain)
    n = size(d.init,1)
    (n*(n+1)) >> 1
end

function pack_lower!(v, d::PSDDomain)
    n = size(d.init,1)
    k = 0
    for j = 1:n
        for i = 1:j-1
            v[k+=1] = -Inf
        end
        v[k+=1] = 0
    end
end
function pack_upper!(v, d::PSDDomain)
    v[:] = +Inf
end
function pack!(v, d::PSDDomain, C::LinAlg.Cholesky)
    @assert C.uplo == 'U'
    U = C.factors
    n = size(d.init,1)
    k = 0
    for j = 1:n
        for i = 1:j-1
            v[k+=1] = U[i,j]
        end
        v[k+=1] = U[j,j]
    end
end
pack!(v, d::PSDDomain, X::PDMats.PDMat) = pack!(v, d, X.chol)
pack!(v, d::PSDDomain, X::AbstractMatrix) = pack!(v, d, cholfact(X))

function unpack(v, d::PSDDomain)
    n = size(d.init,1)
    U = zeros(eltype(v), n,n)
    k = 0
    for j = 1:n
        for i = 1:j-1
            U[i,j] = v[k+=1]
        end
        U[j,j] = v[k+=1]
    end
    PDMat(LinAlg.Cholesky(U, :U))
end


struct ParamSet{T}
    params::T
end
init(p::ParamSet) = map(init, p.params)
packlen(p::ParamSet) = sum(packlen, p.params)

pack_upper(p::ParamSet) = pack_upper!(Array{Float64}(packlen(p)),p)
pack_lower(p::ParamSet) = pack_lower!(Array{Float64}(packlen(p)),p)
pack(p::ParamSet, x)  = pack!(Array{numtype(x)}(packlen(p)), p, x)
pack_init(p::ParamSet)  = pack(p, init(p))

function pack_upper!(v, p::ParamSet)
    k = 0
    for pp in p.params
        n = packlen(pp)
        vv = @view(v[k + (1:n)])
        pack_upper!(vv, pp)
        k += n
    end
    return v
end
function pack_lower!(v, p::ParamSet)
    k = 0
    for pp in p.params
        n = packlen(pp)
        vv = @view(v[k + (1:n)])
        pack_lower!(vv, pp)
        k += n
    end
    return v
end
function pack!(v, p::ParamSet, x)
    k = 0
    for (pp,xx) in zip(p.params,x)
        n = packlen(pp)
        vv = @view(v[k + (1:n)])
        pack!(vv, pp, xx)
        k += n
    end
    return v
end
function unpack(v, p::ParamSet)
    local k::Int
    k = 0
    map(p.params) do pp
        n = packlen(pp)
        vv = @view(v[k + (1:n)])
        k += n
        unpack(vv,pp)
    end
end

