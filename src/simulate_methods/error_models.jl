"""
    Independent(errfn) <: ErrorModel

Assumes the error terms are independent.

`errfn` should be function which takes 2 arguments:
 - `uij`: the value at a particular timepoint of the reduced simulation
 - `θ`  : the population-level parameter vector

Example: proportional error:
```julia
properr(uij, θ) = Normal(uij, uij*θ[end])
```
"""
struct Independent{F} <: ErrorModel
    inderr::F
end

function simulate(m::Independent, θ, ui)
    [rand(m.inderr(uij,θ)) for uij in ui]
end
