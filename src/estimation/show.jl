const _subscriptvector = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
_to_subscript(number) = join([_subscriptvector[parse(Int32, dig)] for dig in string(number)])

function _print_fit_header(io, fpm)
  println(io, string("Successful minimization:",
                     lpad(string(Optim.converged(fpm.optim)), 20)))
  println(io)
  println(io, string("Likelihood approximation:",
                     lpad(typeof(fpm.approx), 19)))
  println(io, string("Objective function value:",
                     lpad(round(Optim.minimum(fpm.optim); sigdigits=round(Int, -log10(DEFAULT_RELTOL))), 19)))
  println(io, string("Total number of observation records:",
                     lpad(sum([length(sub.time) for sub in fpm.data]), 8)))
  println(io, string("Number of active observation records:",
                     lpad(sum(subject -> count(!ismissing, subject.observations.dv), fpm.data),7)))
  println(io, string("Number of subjects:",
                     lpad(length(fpm.data), 25)))
  println(io)
end

function Base.show(io::IO, mime::MIME"text/plain", fpm::FittedPumasModel)
  println(io, "FittedPumasModel\n")
  _print_fit_header(io, fpm)
  # Get all names
  paramnames = []
  paramvals = []
  for (paramname, paramval) in pairs(fpm.param)
    _push_varinfo!(paramnames, paramvals, nothing, nothing, paramname, paramval, nothing, nothing)
  end
  getdecimal = x -> findfirst(c -> c=='.', x)
  maxname = maximum(length, paramnames)
  maxval = max(maximum(length, paramvals), length("Estimate "))
  labels = " "^(maxname+Int(round(maxval/1.2))-3)*"Estimate"
  stringrows = []
  for (name, val) in zip(paramnames, paramvals)
    push!(stringrows, string(name, " "^(maxname-length(name)-getdecimal(val)+Int(round(maxval/1.2))), val, "\n"))
  end
  println(io,"-"^max(length(labels)+1,maximum(length.(stringrows))))
  print(io, labels)
  println(io,"\n" ,"-"^max(length(labels)+1,maximum(length.(stringrows))))
  for stringrow in stringrows
    print(io, stringrow)
  end
  println(io,"-"^max(length(labels)+1,maximum(length.(stringrows))))
end
TreeViews.hastreeview(x::FittedPumasModel) = true
function TreeViews.treelabel(io::IO,x::FittedPumasModel,
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io, mime, Base.Text(Base.summary(x)))
end
function Base.show(io::IO, mime::MIME"text/plain", vfpm::Vector{<:FittedPumasModel})
  println(io, "Vector{<:FittedPumasModel} with $(length(vfpm)) entries\n")
  # _print_fit_header(io, fpm)
  # Get all names
  paramnames = []
  paramvals = []
  paramstds = []

  for (paramname, paramval) in pairs(first(vfpm).param)
    _paramval = [fpm.param[paramname] for fpm in vfpm]
    parammean = mean(_paramval)
    paramstd = parammean*std(_paramval)/100
    _push_varinfo!(paramnames, paramvals, paramstds, nothing, paramname, parammean, paramstd, nothing)
  end
  getdecimal = x -> findfirst(c -> c=='.', x)
  maxname = maximum(length, paramnames)
  maxval = max(maximum(length, paramvals), length("Mean"))
  maxstd = max(maximum(length, paramstds), length("Std"))
  labels = " "^(maxname+Int(round(maxval/1.2))-3)*rpad("Mean", Int(round(maxstd/1.2))+maxval+3)*"Std"
  stringrows = []
  for (name, val, std) in zip(paramnames, paramvals, paramstds)
    push!(stringrows, string(name, " "^(maxname-length(name)-getdecimal(val)+Int(round(maxval/1.2))), val, " "^(maxval-(length(val)-getdecimal(val))-getdecimal(std)+Int(round(maxstd/1.2))), std, "\n"))
  end
  println(io, "Parameter statistics")
  println(io, "-"^max(length(labels)+1,maximum(length.(stringrows))))
  print(io, labels)
  println(io,"\n" ,"-"^max(length(labels)+1,maximum(length.(stringrows))))
  for stringrow in stringrows
    print(io, stringrow)
  end
  println(io,"-"^max(length(labels)+1,maximum(length.(stringrows))))
end
TreeViews.hastreeview(x::Vector{<:FittedPumasModel}) = true
function TreeViews.treelabel(io::IO,x::Vector{<:FittedPumasModel},
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io, mime, Base.Text(Base.summary(x)))
end

function _push_varinfo!(_names, _vals, _rse, _confint, paramname, paramval::PDMat, std, quant)
  mat = paramval.mat
  for j = 1:size(mat, 2)
    for i = j:size(mat, 1)
      # We set stdij to nothing in case RSEs are not requested to avoid indexing
      # into `nothing`.
      stdij = _rse == nothing ? nothing : std[i, j]
      _name = string(paramname)*"$(_to_subscript(i)),$(_to_subscript(j))"
      _push_varinfo!(_names, _vals, _rse, _confint, _name, mat[i, j], stdij, quant)
    end
  end
end
function _push_varinfo!(_names, _vals, _rse, _confint, paramname, paramval::PDiagMat, std, quant)
  mat = paramval.diag
    for i = 1:length(mat)
      # We set stdii to nothing in case RSEs are not requested to avoid indexing
      # into `nothing`.
      stdii = _rse == nothing ? nothing : std.diag[i]
      _name = string(paramname)*"$(_to_subscript(i)),$(_to_subscript(i))"
      _push_varinfo!(_names, _vals, _rse, _confint, _name, mat[i], stdii, quant)
    end
end
_push_varinfo!(_names, _vals, _rse, _confint, paramname, paramval::Diagonal, std, quant) =
  _push_varinfo!(_names, _vals, _rse, _confint, paramname, PDiagMat(paramval.diag), std, quant)
function _push_varinfo!(_names, _vals, _rse, _confint, paramname, paramval::AbstractVector, std, quant)
  for i in 1:length(paramval)
    # We set stdi to nothing in case RSEs are not requested to avoid indexing
    # into `nothing`.
    stdi = _rse == nothing ? nothing : std[i]
    _push_varinfo!(_names, _vals, _rse, _confint, string(paramname, _to_subscript(i)), paramval[i], stdi, quant)
  end
end
function _push_varinfo!(_names, _vals, _rse, _confint, paramname, paramval::Number, std, quant)
  # Push variable names and values. Values are truncated to five significant
  # digits to control width of output.
  push!(_names, string(paramname))
  push!(_vals, string(round(paramval; sigdigits=5)))
  # We only update RSEs and confints if an array was input.
  !(_rse == nothing) && push!(_rse, string(round(100*std/paramval; sigdigits=5)))
  !(_confint == nothing) && push!(_confint, string("[", round(paramval - std*quant; sigdigits=5), ";", round(paramval+std*quant; sigdigits=5), "]"))
end


function Base.show(io::IO, mime::MIME"text/plain", pmi::FittedPumasModelInference)
  fpm = pmi.fpm

  println(io, "FittedPumasModelInference\n")
  _print_fit_header(io, pmi.fpm)

  # Get all names
  standard_errors = stderror(pmi)
  paramnames = []
  paramvals = []
  paramrse = []
  paramconfint = []

  quant = quantile(Normal(), pmi.level + (1.0 - pmi.level)/2)

  for (paramname, paramval) in pairs(fpm.param)
    std = standard_errors[paramname]
    _push_varinfo!(paramnames, paramvals, paramrse, paramconfint, paramname, paramval, std, quant)
  end
  getdecimal = x -> occursin("NaN", x) ? 2 : findfirst(isequal('.'), x)
  getsemicolon = x -> findfirst(c -> c==';', x)
  getafterdec = x -> getsemicolon(x) - getdecimal(x)
  getdecaftersemi = x -> getdecimal(x[getsemicolon(x):end])
  getaftersemdec = x -> occursin("NaN", x) ? length(x) - 1 : findall(isequal('.'), x)[2]
  getafterdecsemi = x -> length(x) - getaftersemdec(x)
  maxname = maximum(length, paramnames)
  maxval = max(maximum(length, paramvals), length("Estimate"))
  maxrs = max(maximum(length, paramrse), length("RSE"))
  maxconfint = max(maximum(length, paramconfint) + 1, length(string(round(pmi.level*100, sigdigits=6))*"% C.I."))
  maxdecconf = maximum(getdecimal, paramconfint)
  maxaftdec = maximum(getafterdec, paramconfint)
  maxdecaftsem = maximum(getdecaftersemi, paramconfint)
  maxaftdecsem = maximum(getafterdecsemi, paramconfint)
  labels = " "^(maxname+Int(round(maxval/1.2))-3)*rpad("Estimate", Int(round(maxrs/1.2))+maxval+3)*rpad("RSE", Int(round(maxconfint/1.2))+maxrs-3)*string(round(pmi.level*100, sigdigits=6))*"% C.I."

  stringrows = []
  for (name, val, rse, confint) in zip(paramnames, paramvals, paramrse, paramconfint)
    confint = string("["," "^(maxdecconf - getdecimal(confint)), confint[2:getsemicolon(confint)-1]," "^(maxaftdec-getafterdec(confint)),"; "," "^(maxdecaftsem - getdecaftersemi(confint)), confint[getsemicolon(confint)+1:end-1], " "^(maxaftdecsem - getafterdecsemi(confint)), "]")
    row = string(name, " "^(maxname-length(name)-getdecimal(val)+Int(round(maxval/1.2))), val, " "^(maxval-(length(val)-getdecimal(val))-getdecimal(rse)+Int(round(maxrs/1.2))), rse, " "^(maxrs-(length(rse)-getdecimal(rse))-getsemicolon(confint)+Int(round(maxconfint/1.2))), confint, "\n")
    push!(stringrows, row)
  end
  println(io, "-"^max(length(labels)+1,length(stringrows[1])))
  print(io, labels)
  println(io, "\n",  "-"^max(length(labels)+1,length(stringrows[1])))
  for stringrow in stringrows
    print(io, stringrow)
  end
  println(io, "-"^max(length(labels)+1,length(stringrows[1])))
end
TreeViews.hastreeview(x::FittedPumasModelInference) = true
function TreeViews.treelabel(io::IO,x::FittedPumasModelInference,
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io,mime,Base.Text(Base.summary(x)))
end

function Base.show(io::IO, mime::MIME"text/plain", pmi::FittedPumasModelInspection)
  println(io, "FittedPumasModelInspection\n")
  println(io, "Fitting was successful: $(Optim.converged(pmi.o.optim))")

  println(io, "Likehood approximations used for")
  println(io, " * Predictions:        $(first(predict(pmi)).approx)")
  println(io, " * Weighted residuals: $(first(wresiduals(pmi)).approx)")
  println(io, " * Empirical bayes:    $(first(empirical_bayes(pmi)).approx)\n")
end
