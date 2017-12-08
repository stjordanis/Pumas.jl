function immediate_absorption_f(t,t0,C0,dose,p,rate)
    Ke = p.CL/p.V
    C0 += dose
    rKe = rate/Ke
    rKe + exp(-(t-t0)*Ke) * (-rKe + C0)
end

function ImmediateAbsorptionModel(tf)
  PKPDAnalyticalProblem{false}(immediate_absorption_f,0.0,(0.0,tf))
end

export ImmediateAbsorptionModel

function one_compartment_f(t,t0,amounts,doses,p,rates)
  Ka = p.Ka
  Ke = p.CL/p.V         # elimination rate
  amt = amounts + doses             # initial values for cmt's + new doses
  Sa = exp(-(t-t0)*Ka)
  Se = exp(-(t-t0)*Ke)
  rKa = rates[1]/Ka
  Depot  = (amt[1] * Sa) + (1-Sa)*rates[1]/(Ka)          # next depot (cmt==1)
  Central =  Ka / (Ka - Ke) * (amt[1] * (Se - Sa) + rates[1]*((1-Se)/Ke - (1-Sa)/Ka)) +
            amt[2] * Se + (1-Se)*rates[2]/Ke # next central (cmt==2)
  @SVector [Depot,Central]
end

function OneCompartmentModel(tf)
  PKPDAnalyticalProblem{false}(one_compartment_f,@SVector([0.0,0.0]),(0.0,tf))
end

export OneCompartmentModel
