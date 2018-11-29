module NCA

using Reexport
using GLM
@reexport using DataFrames

include("utils.jl")
include("type.jl")
include("interpolate.jl")
include("auc.jl")
include("simple.jl")

export NCAdata, showunits
export auc, aumc, lambdaz,
       clast, tlast, cmax, tmax, thalf

for f in (:lambdaz, :cmax, :tmax, :clast, :tlast, :thalf)
  @eval $f(conc, time, args...; kwargs...) = $f(NCAdata(conc, time, args...; kwargs...); kwargs...)
end

end
