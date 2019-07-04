using Pumas, Test
pop = Population(map(i -> Subject(id=i,cvs=(dose=[10,20,30],)),1:3))
poisson_model = @model begin
  @param begin
    tvbase ∈ RealDomain(init=3.0, lower=0.1)
    d50 ∈ RealDomain(init=0.5, lower=0.1)
    Ω  ∈ PSDDomain(fill(0.1, 1, 1))
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    baseline = tvbase*exp(η[1])
  end

  @covariates dose

  @derived begin
    dv ~ @. Poisson(baseline*(1-dose/(dose + d50)))
  end
end

sim = simobs(poisson_model,pop[1])
@test length(sim[:dv]) == 3
