# shooting method to estimate input_rate function
# error function to be optimized
function errfun(C_act, t, r, kel, param)
  f(u,p,t) = r(u, t, param) - kel*u
  c0 = zero(eltype(C_act))
  tspan = (t[1], t[end])
  prob = ODEProblem(f, c0, tspan)
  C = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=t).u
  mse = sum(abs2.(C_act .- C))/length(C)
  mse
end

function calc_input_rate(subj::VivoSubject, r, p0)
  # for now, kel is p[2] incase of bateman model
  kel = subj.pmin[2]
  calc_input_rate(subj.conc, subj.time, r, kel, p0)
end

function calc_input_rate(C_act, t, r, kel, p0, alg=LBFGS(), box=false, lb=nothing, ub=nothing)
  if box == false
    mfit = optimize(p->errfun(C_act, t, r, kel, p), p0, alg)
  else
    if lb == nothing || ub == nothing
      error("lower or upper bound should not be nothing")
    end
    od = OnceDifferentiable(p->errfun(C_act, t, r, kel, p), p0, autodiff=:finite)
    mfit = optimize(od, lb, ub, p0, Fminbox(alg))
  end
  pmin = Optim.minimizer(mfit)
  pmin
end

