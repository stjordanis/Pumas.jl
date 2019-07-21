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
using Pumas, Plots
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

![single_sub](https://user-images.githubusercontent.com/1425562/61312349-3cfbaa00-a7c6-11e9-9777-a3b7c17fbeaa.png)

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
![pop_sim](https://user-images.githubusercontent.com/1425562/61312348-3cfbaa00-a7c6-11e9-9c23-f4bcbfb5930f.png)

Let's roundtrip this simulation to test our estimation routines

```julia
simdf = DataFrame(obs)
simdf.cmt = 1
first(simdf, 6)
```
Read the data in to Pumas

```julia
data = read_pumas(simdf, time=:time,cvs=[:isPM, :wt])
```

Evaluating the results of a model fit goes through an `fit` --> `infer` --> `inspect` --> `validate` cycle

### `fit`

```julia
julia> res = fit(model,data,param,Pumas.FOCEI())
FittedPumasModel

Successful minimization:                true

Likelihood approximation:        Pumas.FOCEI
Objective function value:            8516.61
Total number of observation records:    1210
Number of active observation records:   1210
Number of subjects:                       10

-----------------
       Estimate
-----------------
tvcl    4.4531
tvv    77.547
pmoncl -0.73493
Ω₁,₁    0.037694
Ω₂,₂    0.11752
σ_prop  0.042341
-----------------
```

### `infer`

`infer` provides the model inference


```julia
julia> infer(res)
Calculating: variance-covariance matrix. Done.
FittedPumasModelInference

Successful minimization:                true

Likelihood approximation:        Pumas.FOCEI
Objective function value:            8516.61
Total number of observation records:    1210
Number of active observation records:   1210
Number of subjects:                       10

---------------------------------------------------
       Estimate       RSE           95.0% C.I.
---------------------------------------------------
tvcl    4.4531      9.0087 [ 3.6668   ;  5.2394  ]
tvv    77.547      10.988  [60.846    ; 94.247   ]
pmoncl -0.73493    -4.5534 [-0.80052  ; -0.66934 ]
Ω₁,₁    0.037694   31.359  [ 0.014526 ;  0.060862]
Ω₂,₂    0.11752    52.745  [-0.0039693;  0.23901 ]
σ_prop  0.042341    2.6616 [ 0.040132 ;  0.044549]
---------------------------------------------------
```

### `inspect`

`inspect` gives you the model predictions, residuals and Empirical Bayes estimates

```julia
resout = DataFrame(inspect(res))
```

```julia
julia> first(resout, 6)
6×13 DataFrame
│ Row │ id     │ time    │ isPM  │ wt    │ pred    │ ipred   │ pred_approx │ wres      │ iwres    │ wres_approx │ ebe_1     │ ebe_2     │ ebes_approx │
│     │ String │ Float64 │ Int64 │ Int64 │ Float64 │ Float64 │ Pumas.FOCEI │ Float64   │ Float64  │ Pumas.FOCEI │ Float64   │ Float64   │ Pumas.FOCEI │
├─────┼────────┼─────────┼───────┼───────┼─────────┼─────────┼─────────────┼───────────┼──────────┼─────────────┼───────────┼───────────┼─────────────┤
│ 1   │ 1      │ 0.0     │ 0     │ 68    │ 1326.95 │ 1290.5  │ FOCEI()     │ 0.0141867 │ 0.164838 │ FOCEI()     │ -0.273173 │ 0.0282462 │ FOCEI()     │
│ 2   │ 1      │ 1.0     │ 0     │ 68    │ 1255.42 │ 1236.44 │ FOCEI()     │ 0.247655  │ 0.414528 │ FOCEI()     │ -0.273173 │ 0.0282462 │ FOCEI()     │
│ 3   │ 1      │ 2.0     │ 0     │ 68    │ 1187.56 │ 1184.65 │ FOCEI()     │ -1.44113  │ -1.53356 │ FOCEI()     │ -0.273173 │ 0.0282462 │ FOCEI()     │
│ 4   │ 1      │ 3.0     │ 0     │ 68    │ 1123.17 │ 1135.03 │ FOCEI()     │ -0.66784  │ -1.10145 │ FOCEI()     │ -0.273173 │ 0.0282462 │ FOCEI()     │
│ 5   │ 1      │ 4.0     │ 0     │ 68    │ 1062.1  │ 1087.49 │ FOCEI()     │ -0.67988  │ -1.29264 │ FOCEI()     │ -0.273173 │ 0.0282462 │ FOCEI()     │
│ 6   │ 1      │ 5.0     │ 0     │ 68    │ 1004.17 │ 1041.94 │ FOCEI()     │ 1.14917   │ 0.521982 │ FOCEI()     │ -0.273173 │ 0.0282462 │ FOCEI()     │
```

### `validate` - `vpc`

Finally validate your model with a visual predictive check

```julia
vpc(res,200) |> plot
```
![vpc](https://user-images.githubusercontent.com/1425562/61312346-3cfbaa00-a7c6-11e9-94ef-af2b5c3d2398.png)

or you can do a `vpc` into a new design as well.

Plotting methods on model diagnostics are coming soon.

### Simulate from fitted model

In order to simulate from a fitted model `simobs` can be used. The final parameters of the fitted models are available in the `res.param`

```julia
fitparam = res.param
```

You can then pass these optimized parameters into a `simobs` call and pass the same dataset or simulate into a different design

```
ev_sd_high_dose = DosageRegimen(200, time=0, addl=4, ii=48)
s2 = Subject(id=1,  evs=ev_sd_high_dose, cvs=(isPM=1, wt=70))
```

```julia
obs = simobs(model, s2, fitparam, obstimes=0:1:160)
plot(obs)
```
![highdose](https://user-images.githubusercontent.com/1425562/61313060-a203cf80-a7c7-11e9-8127-8d09ec69c334.png)
```
