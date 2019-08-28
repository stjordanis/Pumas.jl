function to_plottable(subj::Union{VitroForm, UirData}; denseplot = true, plotdensity = 10_000)
  t = subj.time
  start = t[1]; stop = t[end]
  if denseplot
    plott = collect(range(start, stop=stop, length=plotdensity))
  else
    plott = t
  end
  plott, subj.m(plott, subj.pmin)
end

@recipe function f(subj::Union{VitroForm, UirData}; denseplot = true, plotdensity = 10_000)
  t = subj.time
  start = t[1]; stop = t[end]
  if denseplot
    plott = collect(range(start, stop=stop, length=plotdensity))
  else
    plott = t
  end
  plott, subj.m(plott, subj.pmin)
end

@userplot Vitro_plot

@recipe function f(h::Vitro_plot; plotdensity = 10_000, denseplot = true)

  vitro_data = h.args[1]
  
  title  := "Estimated Vitro Model with Original data"
  xlabel := "time"
  ylabel := "Fraction Dissolved (Fdiss)" 
  legend := :bottomright

  for (form, prof) in vitro_data
    @series begin
      seriestype --> :scatter
      label --> "Original " * "$(form)"
      prof.time, prof.conc
    end
  end

  for (form, prof) in vitro_data
    @series begin
      seriestype --> :path
      label --> "Fitted " * "$(form)"
      x, y = to_plottable(prof, plotdensity = plotdensity, denseplot = denseplot)
      x, y
    end
  end
  primary := false
end

