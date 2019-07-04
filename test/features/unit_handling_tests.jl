using Pumas, Test
umodel = @model begin
  @param begin
    k1 = 1.01u"mg"
    I  = 1u"hr"
  end
  @init begin
    x1=k1
  end
  @dynamics begin
    x1'=x1/I
  end
  @derived begin
    C=x1
  end
end
evs = DosageRegimen(45u"mg",cmt=1,time=13u"d")
subject=Subject(id=1,evs=evs)
@test_broken sim=simobs(umodel,subject,obstimes=0u"d":1u"hr":1u"d",alg=OrdinaryDiffEq.Tsit5())
