using Test
using PuMaS, TransformVariables, LinearAlgebra, Distributions

@testset "ParamSets and Domains tests" begin
  p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
                Ω = PSDDomain(2),
                Σ = RealDomain(lower=0.0, init=1.0),
                a = ConstDomain(0.2)))

  t = PuMaS.totransform(p)
  @test TransformVariables.dimension(t) == 8
  u = transform(t, zeros(8))
  @test all(u.θ .> 0)
  @test u.Ω isa PuMaS.PDMats.AbstractPDMat



  pd = ParamSet((θ = Constrained(MvNormal([1.0 0.2; 0.2 1.0]), lower=-2.0),
                 Ω = InverseWishart(13.0, [1.0 0.2; 0.2 1.0])))

  td = PuMaS.totransform(pd)
  @test TransformVariables.dimension(td) == 5
  ud = transform(td, zeros(5))
  @test all(ud.θ .> -2.0)
  @test ud.Ω isa PuMaS.PDMats.AbstractPDMat

  @testset "Promotion" begin
    d = RealDomain(lower=0, upper=1.0)
    @test d.lower === 0.0
    @test d.upper === 1.0
    d = VectorDomain(2,lower=[0  , 2.0], upper=[10  , 4  ], init=[2, 2])
    @test (d.lower...,) == (0.0, 2.0)
    @test (d.upper...,) == (10.0, 4.0)
    @test (d.init...,)  == (2, 2)
  end
end
