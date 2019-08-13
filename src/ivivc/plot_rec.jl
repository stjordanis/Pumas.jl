@recipe function f(subj::Union{VivoForm, UirData}; denseplot = true, plotdensity = 10_000)
  t = subj.time
  start = t[1]; stop = t[end]
  if denseplot
    plott = collect(range(start, stop=stop, length=plotdensity))
  else
    plott = t
  end
  plott, subj.m(plott, subj.pmin)
end
