export RandomEffectSet


function Domain(d::MvNormal)
    n = length(d)
    VectorDomain(fill(-Inf, n), fill(Inf, n), mean(d))
end
function Domain(d::ContinuousUnivariateDistribution)
    RealDomain(minimum(d), maximum(d), median(d))
end

"""
    RandomEffectSet([domains,] dists)

Specifies the `RandomEffect`s:
- `dists` is a collection (e.g. a `NamedTuple`) of distributions of random effect (e.g. `MvNormal(Ω)`)
- `domains` are the corresponding `Domains`.

"""
struct RandomEffectSet{S,T}
    domains::S
    dists::T
end
RandomEffectSet(dists) = RandomEffectSet(map(Domain, dists), dists)

Base.rand(rfx::RandomEffectSet) = map(rand, rfx.dists)


init(rfx::RandomEffectSet) = init(ParamSet(rfx.domains))
packlen(rfx::RandomEffectSet) = packlen(ParamSet(rfx.domains))

pack_upper(rfx::RandomEffectSet) = pack_upper(ParamSet(rfx.domains))
pack_lower(rfx::RandomEffectSet) = pack_lower(ParamSet(rfx.domains))
pack_init(rfx::RandomEffectSet) = pack_init(ParamSet(rfx.domains))
unpack(v, rfx::RandomEffectSet) = unpack(v, ParamSet(rfx.domains))


