# Model dependent methods

## Wagner Nelson method (one compartment model)
function wagner_nelson(c, t, kel)
  sub = NCASubject(c, t)
  intervals = [(t[i], t[i+1]) for i = 1:length(t)-1]
  auc_i = NCA.auc(sub, interval=intervals)
  pushfirst!(auc_i, 0.0)       # area between first to first time point
  c_auc_i = cumsum(auc_i)
  X = c .+ (c_auc_i .* kel)
  X ./ (c_auc_i[end] + c[end]/kel)  # area between time[end] and Inf is c[end]/kel
end

## Loo-Riegelman Method (two compartment model)



# Model independent methods

## Numerical Deconvolution


# shooting method to estimate input_rate function
# error function to be optimized
function errfun(C_act, t, r, kel, param, alg; kwargs...)
  f(u,p,t) = r(u, t, param) - kel*u
  c0 = zero(eltype(C_act))
  tspan = (t[1], t[end])
  prob = ODEProblem(f, c0, tspan)
  C = solve(prob, alg, reltol=1e-8, abstol=1e-8, saveat=t, kwargs...).u
  mse = sum(abs2.(C_act .- C))/length(C)
  mse
end

function calc_input_rate(subj::VivoSubject, r, p0)
  # for now, kel is p[2] incase of bateman model
  kel = subj.pmin[2]
  calc_input_rate(subj.conc, subj.time, r, kel, p0)
end

function calc_input_rate(C_act, t, r, kel, p0; d_alg=Tsit5(),
                          alg=LBFGS(), box=false, lb=nothing, ub=nothing)
  if box == false
    mfit = optimize(p->errfun(C_act, t, r, kel, p, d_alg), p0, alg)
  else
    if lb == nothing || ub == nothing
      error("lower or upper bound should not be nothing")
    end
    od = OnceDifferentiable(p->errfun(C_act, t, r, kel, p, d_alg), p0, autodiff=:finite)
    mfit = optimize(od, lb, ub, p0, Fminbox(alg))
  end
  pmin = Optim.minimizer(mfit)
  pmin
end

