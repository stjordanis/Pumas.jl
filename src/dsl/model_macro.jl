using DataStructures: OrderedDict, OrderedSet
using MacroTools
using ModelingToolkit

islinenum(x) = x isa LineNumberNode
function nt_expr(set, prefix=nothing)
  t = :(())
  for p in set
    sym = prefix === nothing ? p : Symbol(prefix, p)
    push!(t.args, :($p = $sym))
  end
  t
end
function nt_expr(dict::AbstractDict, prefix=nothing)
  t = :(())
  for (p,d) in pairs(dict)
    sym = prefix === nothing ? d : Symbol(prefix, d)
    push!(t.args, :($p = $sym))
  end
  t
end

function var_def(tupvar, indvars)
  quote
    if $tupvar != nothing
      if $tupvar isa NamedTuple
        # Allow for NamedTuples to be in different order
        $(Expr(:block, [:($(esc(v)) = $tupvar.$v) for v in indvars]...))
      else
        $(Expr(:tuple, (esc(v) for v in indvars)...)) = $tupvar
      end
    end
  end
end
var_def(tupvar, inddict::AbstractDict) = var_def(tupvar, keys(inddict))


function extract_params!(vars, params, exprs)
  # should be called on an expression wrapped in a @param
  # expr can be:
  #  a block
  #  a constant expression
  #    a = 1
  #  a domain
  #    a ∈ RealDomain()
  #  a random variable
  #    a ~ Normal()
  if exprs isa Expr && exprs.head == :block
    _exprs = exprs.args
  else
    _exprs = (exprs,)
  end

  for expr in _exprs
    islinenum(expr) && continue
    @assert expr isa Expr
    if expr.head == :call && expr.args[1] in (:∈, :in)
      p = expr.args[2]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      params[p] = expr.args[3]
    elseif expr.head == :(=)
      p = expr.args[1]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      params[p] = :(ConstDomain($(expr.args[2])))
    elseif expr.head == :call && expr.args[1] == :~
      p = expr.args[2]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      params[p] = expr.args[3]
    else
      error("Invalid @param expression: $expr")
    end
  end
end

function param_obj(params)
  :(ParamSet($(esc(nt_expr(params)))))
end

function extract_randoms!(vars, randoms, exprs)
  # should be called on an expression wrapped in a @random
  # expr can be:
  #  a block
  #  a constant expression
  #    a = 1
  #  a random var
  #    a ~ Normal(0,1)

  if exprs isa Expr && exprs.head == :block
    _exprs = exprs.args
  else
    _exprs = (exprs,)
  end

  for expr in _exprs
    islinenum(expr) && continue
    @assert expr isa Expr
    if expr.head == :call && expr.args[1] == :~
      p = expr.args[2]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      randoms[p] = expr.args[3]
      # TODO support const expressions
      # elseif expr.head == :(=)
      #     p = expr.args[1]
      #     p in vars && error("Variable $p already defined")
      #     push!(vars,p)
      #     randoms[p] = :(ConstDomain($(expr.args[2])))
    else
      error("Invalid @random expression: $expr")
    end
  end
end

function random_obj(randoms, params)
  quote
    function (_param)
      $(var_def(:_param, params))
      ParamSet($(esc(nt_expr(randoms))))
    end
  end
end

function extract_syms!(vars, subvars, syms)
  for p in syms
    islinenum(p) && continue
    p in vars && error("Variable $p already defined")
    push!(subvars, p)
    push!(vars, p)
  end
end

function extract_pre!(vars, prevars, exprs)
  # should be called on an expression wrapped in a @pre
  # expr can be:
  #  a block
  #  an assignment
  #    a = 1

  if exprs isa Expr && exprs.head == :block
    _exprs = exprs.args
  else
    _exprs = (exprs,)
  end

  newexpr = Expr(:block)
  for expr in _exprs
    MacroTools.prewalk(expr) do ex
      if ex isa Expr && ((iseq = ex.head == :(=)) || ex.head == :(:=)) && length(ex.args) == 2
        p = ex.args[1]
        p in vars && error("Variable $p already defined")
        if iseq
          push!(vars, p)
          push!(prevars, p)
        end
        push!(newexpr.args, :($p = $(ex.args[2])))
      else
        ex
      end
    end
  end
  newexpr
end

function pre_obj(preexpr, prevars, params, randoms, covariates)
  quote
    function (_param, _random, _covariates)
      $(var_def(:_param, params))
      $(var_def(:_random, randoms))
      $(var_def(:_covariates, covariates))
      $(esc(preexpr))
      $(esc(nt_expr(prevars)))
    end
  end
end


function extract_dynamics!(vars, odevars, ode_init, expr::Expr, eqs)
  if expr.head == :block
    for ex in expr.args
      islinenum(ex) && continue
      extract_dynamics!(vars, odevars, ode_init, ex, eqs)
    end
  elseif expr.head == :(=) || expr.head == :(:=)
    dp = expr.args[1]
    isder = dp isa Expr && dp.head == Symbol('\'')
    #dp in vars && error("Variable $dp already defined")
    (isder && length(dp.args) == 1) ||
      error("Invalid variable $dp: must be in the form of X'")
    p = dp.args[1]
    lhs = :(___D($p))
    push!(eqs.args, :($lhs ~ $(expr.args[2])))
    if p ∈ keys(ode_init)
      push!(odevars, p)
    elseif p ∉ vars
      push!(vars,p)
      push!(odevars, p)
      ode_init[p] = 0.0
    end
  else
    error("Invalid @dynamics expression: $expr")
  end
  return true # isstatic
end

# used for pre-defined analytical systems
function extract_dynamics!(vars, odevars, ode_init, sym::Symbol, eqs)
  obj = eval(sym)
  if obj isa Type && obj <: ExplicitModel # explict model
    for p in varnames(obj)
      if p in keys(ode_init)
        push!(odevars, p)
      else
        if p ∉ vars
          push!(vars,p)
          push!(odevars, p)
        end
        ode_init[p] = 0
      end
    end
    return true # isstatic
  else
    # we assume they are defined in @init
    for p in keys(ode_init)
      push!(odevars, p)
    end
    return false # isstatic
  end
end

function init_obj(ode_init,odevars,prevars,isstatic)
  if isstatic
    vecexpr = []
    for p in odevars
      push!(vecexpr, ode_init[p])
    end
    uType = SLArray{Tuple{length(odevars)},(odevars...,)}
    typeexpr = :($uType())
    append!(typeexpr.args,vecexpr)
    quote
      function (_pre,t)
        $(var_def(:_pre, prevars))
        $(esc(typeexpr))
      end
    end
  else
    vecexpr = :([])
    for p in odevars
      push!(vecexpr.args, ode_init[p])
    end
    quote
      function (_pre,t)
        $(var_def(:_pre, prevars))
        $(esc(vecexpr))
      end
    end
  end
end

function dynamics_obj(odeexpr::Expr, pre, odevars, bvars, eqs, isstatic)
  odeexpr == :() && return nothing
  dvars = :(begin end)
  params = :(begin end)
  version = isstatic ? ModelingToolkit.SArrayFunction : ModelingToolkit.ArrayFunction
  diffeq = :(ODEProblem(ODEFunction(DiffEqSystem($eqs, t, [], Variable[]),version=$version),nothing,nothing,nothing))

  # DVar
  for v in odevars
    push!(dvars.args, :($v = Variable($(Meta.quot(v)), [t])))
    push!(diffeq.args[2].args[2].args[4].args, v)
  end
  # Param
  for p in pre
    push!(params.args, :($p = Variable($(Meta.quot(p)); known = true)))
    push!(diffeq.args[2].args[2].args[5].args, p)
  end
  quote
    let
      t = Variable(:t; known = true)
      $dvars
      ___D = Differential(t)
      $params
      $bvars
      $diffeq
    end
  end
end
function dynamics_obj(odename::Symbol, pre, odevars, bvars, eqs, isstatic)
  quote
    ($odename isa Type && $odename <: ExplicitModel) ? $odename() : $odename
  end
end


function extract_defs!(vars, defsdict, exprs)
  # should be called on an expression wrapped in a @derived
  # expr can be:
  #  a block
  #  an assignment
  #    a = 1

  if exprs isa Expr && exprs.head == :block
    _exprs = exprs.args
  else
    _exprs = (exprs,)
  end

  for expr in _exprs
    islinenum(expr) && continue
    @assert expr isa Expr
    if expr.head == :(=)
      p = expr.args[1]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      defsdict[p] = expr.args[2]
    else
      error("Invalid expression: $expr")
    end
  end
end

function extract_randvars!(vars, randvars, postexpr, expr)
  @assert expr isa Expr
  if expr.head == :block
    for ex in expr.args
      islinenum(ex) && continue
      extract_randvars!(vars, randvars, postexpr, ex)
    end
  elseif ((iseq = expr.head == :(=)) || expr.head == :(:=)) && length(expr.args) == 2
    p = expr.args[1]
    if p ∉ vars && iseq
      push!(vars,p)
      push!(randvars,p)
    end
    push!(postexpr.args, :($p = $(expr.args[2])))
  elseif expr isa Expr && expr.head == :call && expr.args[1] == :~ && length(expr.args) == 3
    p = expr.args[2]
    if p ∉ vars
      push!(vars,p)
      push!(randvars,p)
    end
    push!(postexpr.args, :($p = $(expr.args[3])))
  else
    error("Invalid expression: $expr")
  end
end

function bvar_def(collection, indvars)
  quote
    if $collection isa DataFrame
      $(Expr(:block, [:($(esc(v)) = $collection.$v) for v in indvars]...))
    elseif $collection isa DESolution
      $(Expr(:block, [:($(esc(v)) = map(x -> x[$i], $collection.u)) for (i,v) in enumerate(indvars)]...))
    else eltype($collection) <: SArray
      $(Expr(:block, [:($(esc(v)) = map(x -> x[$i], $collection)) for (i,v) in enumerate(indvars)]...))
      #else
      #$(Expr(:block, [:($(esc(v)) = map(x -> x.$v, $collection)) for v in indvars]...))
    end
  end
end

function derived_obj(derivedexpr, derivedvars, pre, odevars)
  quote
    function (_pre,_sol,_obstimes)
      $(var_def(:_pre, pre))
      if _sol != nothing
        $(bvar_def(:(_sol), odevars))
      end
      $(esc(:t)) = _obstimes
      $(esc(derivedexpr))
      $(esc(nt_expr(derivedvars)))
    end
  end
end

function observed_obj(observedexpr, observedvars, pre, odevars, derivedvars)
  quote
    function (_pre,_sol,_obstimes,_samples)
      $(var_def(:_pre, pre))
      $(var_def(:_samples, derivedvars))
      if _sol != nothing
        $(bvar_def(:(_sol), odevars))
      end
      $(esc(:t)) = _obstimes
      $(esc(observedexpr))
      $(esc(nt_expr(observedvars)))
    end
  end
end

function broadcasted_vars(vars)
  foreach(vars.args) do ex
    isa(ex, Expr) || return
    ex.args[2] = :(@. $(ex.args[2]))
  end
  return vars
end

function add_vars(ex::Expr, vars)
  args = ex.head === :block ? ex.args : [ex]
  return Expr(:block, [vars.args; args]...)
end

macro model(expr)

  vars = Set{Symbol}([:t]) # t is the only reserved symbol
  params = OrderedDict{Symbol, Any}()
  randoms = OrderedDict{Symbol, Any}()
  covariates = OrderedSet{Symbol}()
  prevars  = OrderedSet{Symbol}()
  ode_init  = OrderedDict{Symbol, Any}()
  odevars  = OrderedSet{Symbol}()
  preexpr = :()
  derivedvars = OrderedSet{Symbol}()
  eqs = Expr(:vect)
  derivedvars = OrderedSet{Symbol}()
  derivedexpr = Expr(:block)
  observedvars = OrderedSet{Symbol}()
  observedexpr = Expr(:block)
  bvars = :(begin end)
  local vars, params, randoms, covariates, prevars, preexpr, odeexpr, odevars
  local ode_init, eqs, derivedexpr, derivedvars, observedvars, observedexpr
  local isstatic, bvars

  isstatic = true
  odeexpr = :()
  lnndict = Dict{Symbol,LineNumberNode}()

  for ex ∈ expr.args
    islinenum(ex) && continue
    (isa(ex, Expr) && ex.head === :macrocall) || throw(ArgumentError("expected macro call, got $ex"))

    if ex.args[1] == Symbol("@param")
      extract_params!(vars, params, ex.args[3])
    elseif ex.args[1] == Symbol("@random")
      extract_randoms!(vars, randoms, ex.args[3])
    elseif ex.args[1] == Symbol("@covariates")
      extract_syms!(vars, covariates, ex.args[3:end])
    elseif ex.args[1] == Symbol("@pre")
      preexpr = extract_pre!(vars,prevars,ex.args[3])
    elseif ex.args[1] == Symbol("@vars")
      bvars = broadcasted_vars(ex.args[3])
    elseif ex.args[1] == Symbol("@init")
      extract_defs!(vars,ode_init, add_vars(ex.args[3], bvars))
    elseif ex.args[1] == Symbol("@dynamics")
      isstatic = extract_dynamics!(vars, odevars, ode_init, ex.args[3], eqs)
      odeexpr = ex.args[3]
    elseif ex.args[1] == Symbol("@derived")
      extract_randvars!(vars, derivedvars, derivedexpr, add_vars(ex.args[3], bvars))
      observedvars = copy(derivedvars)
    elseif ex.args[1] == Symbol("@observed")
      extract_randvars!(vars, observedvars, observedexpr, add_vars(ex.args[3], bvars))
    else
      throw(ArgumentError("Invalid macro $(ex.args[1])"))
    end
    #return nothing
  end

  prevars = union(prevars, keys(params), keys(randoms), covariates)

  ex = quote
    x = PKPDModel(
    $(param_obj(params)),
    $(random_obj(randoms,params)),
    $(pre_obj(preexpr,prevars,params,randoms,covariates)),
    $(init_obj(ode_init,odevars,prevars,isstatic)),
    $(dynamics_obj(odeexpr,prevars,odevars,bvars,eqs,isstatic)),
    $(derived_obj(derivedexpr,derivedvars,prevars,odevars)),
    $(observed_obj(observedexpr,observedvars,prevars,odevars,derivedvars)))
    function Base.show(io::IO, ::typeof(x))
      println(io,"PKPDModel")
      println(io,"  Parameters: ",$(join(keys(params),", ")))
      println(io,"  Random effects: ",$(join(keys(randoms),", ")))
      println(io,"  Covariates: ",$(join(covariates,", ")))
      println(io,"  Dynamical variables: ",$(join(odevars,", ")))
      println(io,"  Derived: ",$(join(derivedvars,", ")))
      println(io,"  Observed: ",$(join(observedvars,", ")))
    end
    x
  end
  ex.args[1] = __source__
  ex
end
