# Pumas.jl

[![Build Status](https://gitlab.com/PumasAI/Pumas-jl/badges/master/build.svg)](https://gitlab.com/PumasAI/Pumas-jl/badges/master/build.svg)
[![codecov](https://codecov.io/gh/PumasAI/Pumas.jl/branch/master/graph/badge.svg?token=O3F3YVonX8)](https://codecov.io/gh/PumasAI/Pumas.jl)

Pumas: A Pharmaceutical Modeling and Simulation toolkit

## Resources
  * [Downloads & Install Instructions](https://pumas.ai/download)
  * [Documentation](https://pumas.ai/documentation)
  * [Tutorials](https://pumas.ai/tutorials)
  * [Blog](https://pumas.ai/blog)
  * [Discourse Forum](https://discourse.pumas.ai/)

## A simple PK model

```julia
using Pumas, Plots, Queryverse
```

A simple one compartment oral absorption model using an analytical solution

```julia
model = @model begin
  @param   begin
    tvcl ∈ RealDomain(lower=0, init = 4.0)
    tvv ∈ RealDomain(lower=0, init = 70)
    pmoncl ∈ RealDomain(lower = -0.99, init= -0.7)
    Ω ∈ PDiagDomain(init=[0.09,0.09])
    σ_prop ∈ RealDomain(lower=0,init=0.04)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = tvcl * (1 + pmoncl*isPM) * (wt/70)^0.75 * exp(η[1])
    V  = tvv * (wt/70) * exp(η[2])
  end
  @covariates wt isPM

  @dynamics ImmediateAbsorptionModel
    #@dynamics begin
    #    Central' =  - (CL/V)*Central
    #end

  @derived begin
      cp = @. 1000*(Central / V)
      dv ~ @. Normal(cp, sqrt(cp^2*σ_prop))
    end
end
```

Develop a simple dosing regimen for a subject

```julia
ev = DosageRegimen(100, time=0, addl=4, ii=24)
s1 = Subject(id=1,  evs=ev, cvs=(isPM=1, wt=70))
```

Simulate a plasma concentration time profile

```julia
param = init_param(model)
obs = simobs(model, s1, param, obstimes=0:1:120)
plot(obs)
```

Generate a population of subjects

```julia
choose_covariates() = (isPM = rand([1, 0]),
              wt = rand(55:80))
pop_with_covariates = Population(map(i -> Subject(id=i, evs=ev, cvs=choose_covariates()),1:10))
```

Simulate into the population

```julia
obs = simobs(model, pop_with_covariates, param, obstimes=0:1:120)
```
and visualize the output

```julia
plot(obs)
```

Let's roundtrip this simulation to test our estimation routines

```julia
simdf = DataFrame(obs) |>
  @mutate(cmt = 1) |>
  DataFrame
first(simdf, 6)
```
Read the data in to Pumas

```julia
data = read_pumas(simdf, time=:time,cvs=[:isPM, :wt])
res = fit(model,data,param,Pumas.FOCEI())
```

Evaluating the results of a model fit goes through an `fit` --> `infer` --> `inspect` --> `validate` cycle

just calling `res` will print out the parameter estimates

```julia
res
```

`infer` provides the model inference 
```julia
infer(res)
```

```julia
resout = DataFrame(inspect(res))
```
