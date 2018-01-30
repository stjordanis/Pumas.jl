export RandomEffect, RandomEffectSet

struct RandomEffect{S,T}
    domain::S
    dist::T
end

function RandomEffect(d::MvNormal)
    n = length(d)
    RandomEffect(VectorDomain(fill(-Inf, n), fill(Inf, n), mean(d)), d)
end
function RandomEffect(d::ContinuousUnivariateDistribution)
    RandomEffect(RealDomain(minimum(d), maximum(d), median(d)), d)
end

struct RandomEffectSet{T}
    effects::T
end

# struct ConstDist{T} <: Distribution
#     val::T
# end
# function RandomEffect(d::ConstDist)
#     RandomEffect(ConstDomain(d.val), d)
# end

Base.rand(rfx::RandomEffectSet) = map(rf -> rand(rf.dist), rfx.effects)


## define a conversion to simplify packing/unpacking
ParamSet(rfx::RandomEffectSet) = ParamSet(map(rf -> rf.domain, rfx.effects))

init(rfx::RandomEffectSet) = init(ParamSet(rfx))
packlen(rfx::RandomEffectSet) = packlen(ParamSet(rfx))

pack_upper!(v, rfx::RandomEffectSet) = pack_upper!(v, ParamSet(rfx))
pack_lower!(v, rfx::RandomEffectSet) = pack_lower!(v, ParamSet(rfx))
pack!(v, rfx::RandomEffectSet, x) = pack!(v, ParamSet(rfx), x)
unpack(v, rfx::RandomEffectSet) = unpack(v, ParamSet(rfx))


