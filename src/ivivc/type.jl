# IVIVCSubject
mutable struct VivoForm{ID, C, T, F, D}
  id::ID
  conc::C
  time::T
  form::F
  dose::D
  function VivoForm(conc, time, form, dose, id=1)
    return new{typeof(id), typeof(conc), typeof(time),
                typeof(form), typeof(dose)}(id, conc, time, form, dose)
  end
end

Base.Broadcast.broadcastable(q::VivoForm) = Ref(q)

function Base.show(io::IO, n::VivoForm)
  df = DataFrame(id = n.id, time = n.time, conc = n.conc, formulation = n.form, dose = n.dose)
  show(io::IO, df)
end

# VivoData
struct VivoData{popType} <: Ivivc
  subjects::popType
  function VivoData(_pop)
    # TODO: can do some checking and other stuff
    return new{typeof(_pop)}(_pop)
  end
end

# VitroData
mutable struct VitroData{ID, C, T, F, pType} <: Ivivc
  id::ID
  conc::C
  time::T
  form::F
  # params which are needed for modeling
  m::Function                           # model type
  alg::Optim.FirstOrderOptimizer        # alg to optimize cost function
  p0::pType                             # intial values of params
  ub::pType                             # upper bound of params
  lb::pType                             # lower bound of params
  pmin::pType                           # optimized params
  function VitroData(conc, time, form, id=1)
    return new{typeof(id), typeof(conc), typeof(time),
                typeof(form), typeof(conc)}(id, conc, time, form)
  end
end

Base.Broadcast.broadcastable(q::VitroData) = Ref(q)

function Base.show(io::IO, n::VitroData)
  df = DataFrame(id = n.id, time = n.time, conc = n.conc, formulation = n.form)
  show(io::IO, df)
end

# VitroData
struct VitroData{popType} <: Ivivc
  subjects::popType
  function VitroData(_pop)
    # TODO: can do some checking and other stuff
    return new{typeof(_pop)}(_pop)
  end
end

# UirData
struct UirData{C, T, F, D, pType}
  conc::C
  time::T
  form::F
  dose::D
  # params which are needed for modeling
  m::Function                           # model type
  alg::Optim.FirstOrderOptimizer        # alg to optimize cost function
  p0::pType                             # intial values of params
  ub::Union{Nothing, pType}             # upper bound of params
  lb::Union{Nothing, pType}             # lower bound of params
  pmin::pType                           # optimized params
  function UirData(conc, time, form, dose)
    return new{typeof(conc), typeof(time),
                typeof(form), typeof(dose), typeof(conc)}(conc, time, form, dose)
  end

Base.Broadcast.broadcastable(q::VitroData) = Ref(q)

function Base.show(io::IO, n::UirData)
  df = DataFrame(time = n.time, conc = n.conc, formulation = n.form, dose=n.dose)
  show(io::IO, df)
end
