# IVIVCSubject
mutable struct VivoSubject{ID, C, T, F, D}
  id::ID
  conc::C
  time::T
  form::F
  dose::D
  function VivoSubject(conc, time, form, dose, id=1)
    return new{typeof(id), typeof(conc), typeof(time),
                typeof(form), typeof(dose)}(id, conc, time, form, dose)
  end
end

Base.Broadcast.broadcastable(q::VivoSubject) = Ref(q)
function Base.show(io::IO, n::VivoSubject)
  println(io, "VivoSubject:")
  df = DataFrame(id = n.id, time = n.time, conc = n.conc, form = n.form, dose = n.dose)
  show(io::IO, df)
  # println(io, "  id:          $(n.id)")
  # println(io, "  conc:         $(n.conc)")
  # println(io, "  time:         $(n.time)")
  # println(io, "  formulation:  $(n.form)")
  # println(io, "  dose:         $(n.dose)")
end

# VivoPopulation
struct VivoPopulation{T<:VivoSubject} <: AbstractVector{T}
  subjects::Vector{T}
  function VivoPopulation(_pop::Vector{<:VivoSubject})
    E = eltype(_pop)
    # TODO: can do some checking and other stuff
    return new{E}(_pop)
  end
end

# VitroSubject
mutable struct VitroSubject{ID, C, T, F, M, pType}
  id::ID
  conc::C
  time::T
  form::F
  # params which are needed for modeling
  m::Function          # model type
  alg::Optim.FirstOrderOptimizer        # alg to optimize cost function
  p0::pType            # intial values of params
  ub::pType            # upper bound of params
  lb::pType            # lower bound of params
  pmin::pType          # optimized params
  function VitroSubject(conc, time, form, id=1)
    p0 = zero(conc); lb = zero()
    return new{typeof(id), typeof(conc), typeof(time),
                typeof(form), typeof(one(conc))}(id, conc, time, form)
  end
end

Base.Broadcast.broadcastable(q::VitroSubject) = Ref(q)
function Base.show(io::IO, n::VitroSubject)
  println(io, "VitroSubject:")
  df = DataFrame(id = n.id, time = n.time, conc = n.conc, form = n.form)
  show(io::IO, df)
  # println(io, "  id:          $(n.id)")
  # println(io, "  conc:         $(n.conc)")
  # println(io, "  time:         $(n.time)")
  # println(io, "  formulation:  $(n.form)")
end

# VitroPopulation
struct VitroPopulation{T<:VitroSubject} <: AbstractVector{T}
  subjects::Vector{T}
  function VitroPopulation(_pop::Vector{<:VitroSubject})
    E = eltype(_pop)
    # TODO: can do some checking and other stuff
    return new{E}(_pop)
  end
end  