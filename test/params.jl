using Test
using PuMaS, TransformVariables, LinearAlgebra, Distributions

p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
              Ω = PSDDomain(2),
              Σ = RealDomain(lower=0.0, init=1.0),
              a = ConstDomain(0.2)))

t = PuMaS.totransform(p)
@test TransformVariables.dimension(t) == 8
u = transform(t, zeros(8))
@test all(u.θ .> 0)
@test u.Ω isa Cholesky



pd = ParamSet((θ = Constrained(MvNormal([1.0 0.2; 0.2 1.0]), lower=-2.0),
              Ω = InverseWishart(13.0, [1.0 0.2; 0.2 1.0])))

td = PuMaS.totransform(pd)
@test TransformVariables.dimension(td) == 5
ud = transform(td, zeros(5))
@test all(ud.θ .> -2.0)
@test ud.Ω isa Cholesky

