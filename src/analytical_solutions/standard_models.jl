export ImmediateAbsorptionModel, OneCompartmentModel, OneCompartmentParallelModel

abstract type ExplicitModel end

struct ImmediateAbsorptionModel <: ExplicitModel end
function (::ImmediateAbsorptionModel)(t,t0,C0,dose,p,rate)
  Ke = p.CL/p.V
  C0 += dose
  rKe = rate/Ke
  rKe + exp(-(t-t0)*Ke) * (-rKe + C0)
end
varnames(::Type{ImmediateAbsorptionModel}) = [:Central]
pk_init(::ImmediateAbsorptionModel) = SLVector(Central=0.0)

struct OneCompartmentModel <: ExplicitModel end
function (::OneCompartmentModel)(t,t0,amounts,doses,p,rates)
  Ka = p.Ka
  Ke = p.CL/p.V           # elimination rate
  amt = amounts + doses   # initial values for cmt's + new doses
  Sa = exp(-(t-t0)*Ka)
  Se = exp(-(t-t0)*Ke)
  rKa = rates[1]/Ka
  Depot  = (amt[1] * Sa) + (1-Sa)*rates[1]/(Ka)          # next depot (cmt==1)
  Central =  Ka / (Ka - Ke) * (amt[1] * (Se - Sa) + rates[1]*((1-Se)/Ke - (1-Sa)/Ka)) +
    amt[2] * Se + (1-Se)*rates[2]/Ke # next central (cmt==2)

  return LabelledArrays.SLVector(Depot=Depot, Central=Central)
end
varnames(::Type{OneCompartmentModel}) = [:Depot, :Central]
pk_init(::OneCompartmentModel) = SLVector(Depot=0.0,Central=0.0)

OneCompartmentParallelVector = @SLVector (:Depot1, :Depot2, :Central)

struct OneCompartmentParallelModel <: ExplicitModel end
function (::OneCompartmentParallelModel)(t,t0,amounts,doses,p,rates)
  ka1 = p.Ka1
  ka2 = p.Ka2
  CL = p.CL
  V = p.V
  ke = CL/V         # elimination rate
  amt = amounts + doses  # initial
  Sa1 = exp(-(t-t0)*ka1)
  Sa2 = exp(-(t-t0)*ka2)
  Se = exp(-(t-t0)*ke)

  Depot1  = amt[1] * Sa1 + rates[1]/ka1*(1-Sa1)          # next depot1 (cmt==1)

  Depot2  = amt[2] * Sa2 + rates[2]/ka2*(1-Sa2)          # next depot2 (cmt==2)

  Central =  ka1 / (ka1 - ke) * (amt[1] * (Se - Sa1) + rates[1]*((1-Se)/ke - (1-Sa1)/ka1)) +
  ka2 / (ka2 - ke) * (amt[2] * (Se - Sa2) + rates[2]*((1-Se)/ke - (1-Sa2)/ka2)) +
  amt[3] * Se + rates[3]/ke*(1-Se) # next central (cmt==3)
  OneCompartmentParallelVector(Depot1,Depot2,Central)
end
pk_init(::OneCompartmentParallelModel) = SLVector(Depot1=0.0,Depot2=0.0,Central=0.0)

varnames(::Type{OneCompartmentParallelModel}) = [:Depot1, :Depot2, :Central]
