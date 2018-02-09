using DataStructures
using MacroTools
using ParameterizedFunctions

export @model

if VERSION < v"0.7-"
    islinenum(x) = x isa Expr && x.head == :line
    function nt_expr(set)
        t = :(NamedTuples.@NT)
        for p in set
            push!(t.args, :($p = $p))
        end
        t
    end
    function nt_expr(dict::Associative)
        t = :(NamedTuples.@NT)
        for (p,d) in dict
            push!(t.args, :($p = $d))
        end
        t
    end
else
    islinenum(x) = x isa LineNumberNode
    function nt_expr(set)
        t = :(())
        for p in set
            push!(t.args, :($p = $p))
        end
        t
    end
    function nt_expr(dict::Associative)
        t = :(())
        for (p,d) in pairs(dict)
            push!(t.args, :($p = $d))
        end
        t
    end
end

function var_def(tupvar, indvars)
    quote
        if $tupvar != nothing
            $(Expr(:tuple, map(esc,indvars)...)) = $tupvar
        end
    end
end
var_def(tupvar, inddict::Associative) = var_def(tupvar, keys(inddict))


function extract_params!(vars, params, exprs...)
    # should be called on an expression wrapped in a @param
    # expr can be:
    #  a block
    #  a constant expression
    #    a = 1
    #  a domain
    #    a ∈ RealDomain()
    for expr in exprs
        @assert expr isa Expr
        if expr.head == :block
            for ex in expr.args
                if !islinenum(ex)
                    extract_params!(vars, params, ex)
                end
            end
        elseif expr.head == :call && expr.args[1] in (:∈, :in)
            p = expr.args[2]
            p in vars && error("Variable $p already defined")
            push!(vars,p)
            params[p] = expr.args[3]
        elseif expr.head == :(=)
            p = expr.args[1]
            p in vars && error("Variable $p already defined")
            push!(vars,p)
            params[p] = :(ConstDomain($(expr.args[2])))
        else
            error("Invalid @param expression: $expr")
        end
    end
end

function param_obj(params)
    :(ParamSet($(esc(nt_expr(params)))))
end

function extract_randoms!(vars, randoms, exprs...)
    # should be called on an expression wrapped in a @random
    # expr can be:
    #  a block
    #  a constant expression
    #    a = 1
    #  a random var
    #    a ~ Normal(0,1)
    for expr in exprs
        @assert expr isa Expr
        if expr.head == :block
            for ex in expr.args
                islinenum(ex) && continue
                extract_randoms!(vars,randoms, ex)
            end
        elseif expr.head == :call && expr.args[1] == :~
            p = expr.args[2]
            p in vars && error("Variable $p already defined")
            push!(vars,p)
            randoms[p] = :(RandomEffect($(expr.args[3])))
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
            RandomEffectSet($(esc(nt_expr(randoms))))
        end
    end
end

function extract_syms!(vars, dict, syms)
    for p in syms
        p in vars && error("Variable $p already defined")
        dict[p] = true
    end
end

function extract_collate!(vars, collate, exprs...)
    # should be called on an expression wrapped in a @collate
    # expr can be:
    #  a block
    #  an assignment
    #    a = 1
    for expr in exprs
        @assert expr isa Expr
        if expr.head == :block
            for ex in expr.args
                islinenum(ex) && continue
                extract_collate!(vars, collate, ex)
            end
        elseif expr.head == :(=)
            p = expr.args[1]
            p in vars && error("Variable $p already defined")
            push!(vars,p)
            collate[p] = expr.args[2]
        else
            error("Invalid @collate expression: $expr")
        end
    end
end

function collate_obj(collate, params, randoms, data_cov)
    quote
        function (_param, _random, _data_cov)
            $(var_def(:_param, params))
            $(var_def(:_random, randoms))
            $(var_def(:_data_cov, data_cov))
            $(esc(nt_expr(collate)))
        end
    end
end


function extract_dynamics!(vars, odevars, expr::Expr)
    if expr.head == :block
        for ex in expr.args
            islinenum(ex) && continue
            extract_dynamics!(vars, odevars, ex)
        end
    elseif expr.head == :(=)
        dp = expr.args[1]
        s = string(dp)
        startswith(s, 'd') && length(s) > 1 || error("Invalid variable $s: must be prefixed with 'd'")
        dp in vars && error("Variable $dp already defined")
        p = Symbol(s[2:end])
        p in vars && error("Variable $p already defined")
        push!(vars,p)
        push!(odevars, p)
    else
        error("Invalid @dynamics expression: $expr")
    end
end

# used for pre-defined analytical systems
function extract_dynamics!(vars, odevars, sym::Symbol)
    pp = varnames(eval(sym))
    for p in pp
        p in vars && error("Variable $p already defined")
        push!(vars,p)
        push!(odevars, p)
    end
end


# here we just use the ParameterizedFunctions @ode_def
function dynamics_obj(odeexpr::Expr, collate, odevars)
    quote
        raw_pkpd_problem(ParameterizedFunctions.@ode_def($(esc(:FooBar)), $(esc(odeexpr)), $(map(esc,keys(collate))...)), zeros($(length(odevars))))
    end
end
function dynamics_obj(odename::Symbol, collate, odevars)
    quote
        $odename()
    end
end


function extract_post!(vars, post, exprs...)
    # should be called on an expression wrapped in a @post
    # expr can be:
    #  a block
    #  an assignment
    #    a = 1
    for expr in exprs
        @assert expr isa Expr
        if expr.head == :block
            for ex in expr.args
                islinenum(ex) && continue
                extract_post!(vars, post, ex)
            end
        elseif expr.head == :(=)
            p = expr.args[1]
            p in vars && error("Variable $p already defined")
            push!(vars,p)
            post[p] = expr.args[2]
        else
            error("Invalid @post expression: $expr")
        end
    end
end

function post_obj(post, params, randoms, data_cov, collate, odevars)
    quote
        function (_param, _random, _data_cov,_collate,_odevars,t)
            $(var_def(:_param, params))
            $(var_def(:_random, randoms))
            $(var_def(:_data_cov, data_cov))
            $(var_def(:_collate, collate))
            $(var_def(:_odevars, odevars))
            $(esc(nt_expr(post)))
        end
    end
end

function extract_randvars!(vars, randvars, expr)
    MacroTools.prewalk(expr) do ex
        if ex isa Expr && ex.head == :call && ex.args[1] == :~ && length(ex.args) == 3
            p = ex.args[2]
            p in vars && error("Variable $p already defined")
            push!(vars,p)
            push!(randvars,p)
            :($p = $(ex.args[3]))
        else
            ex
        end
    end
end


function error_obj(errorexpr, errorvars, params, randoms, data_cov, collate, odevars)
    quote
        function (_param, _random, _data_cov, _collate, _odevars, t)
            $(var_def(:_param, params))
            $(var_def(:_random, randoms))
            $(var_def(:_data_cov, data_cov))
            $(var_def(:_collate, collate))
            $(var_def(:_odevars, odevars))
            $(esc(errorexpr))
            $(esc(nt_expr(errorvars)))
        end
    end
end


macro model(expr)

    vars = Set{Symbol}([:t]) # t is the only reserved symbol
    params = OrderedDict{Symbol, Any}()
    randoms = OrderedDict{Symbol, Any}()
    data_cov = OrderedDict{Symbol, Any}()
    collate  = OrderedDict{Symbol, Any}()
    post  = OrderedDict{Symbol, Any}()
    odevars  = OrderedSet{Symbol}()
    errorvars  = OrderedSet{Symbol}()
    errorexpr = :()
    local vars, params, randoms, data_cov, collate, post, odeexpr, odevars, errorexpr, errorvars

    MacroTools.prewalk(expr) do ex
        ex isa Expr && ex.head == :block && return ex
        islinenum(ex) && return nothing
        @assert ex isa Expr && ex.head == :macrocall
        if ex.args[1] == Symbol("@param")
            extract_params!(vars, params, ex.args[2:end]...)
        elseif ex.args[1] == Symbol("@random")
            extract_randoms!(vars, randoms, ex.args[2:end]...)
        elseif ex.args[1] == Symbol("@data_cov")
            extract_syms!(vars, data_cov, ex.args[2:end])
        elseif ex.args[1] == Symbol("@collate")
            extract_collate!(vars,collate, ex.args[2:end]...)
        elseif ex.args[1] == Symbol("@dynamics")
            extract_dynamics!(vars, odevars, ex.args[2])
            odeexpr = ex.args[2]
        elseif ex.args[1] == Symbol("@post")
            extract_post!(vars,post, ex.args[2:end]...)
        elseif ex.args[1] == Symbol("@error")
            errorexpr = extract_randvars!(vars, errorvars, ex.args[2])
        else
            error("Invalid macro $(ex.args[1])")
        end
        return nothing
    end

    quote
        PKPDModel(
            $(param_obj(params)),
            $(random_obj(randoms,params)),
            $(collate_obj(collate,params,randoms,data_cov)),
            $(dynamics_obj(odeexpr,collate,odevars)),
            $(post_obj(post,params,randoms,data_cov,collate, odevars)),
            $(error_obj(errorexpr, errorvars, params, randoms, data_cov, collate, odevars)))
    end
end
