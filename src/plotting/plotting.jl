################################################################################
#                           Useful utility functions                           #
################################################################################

# heuristic to determine if a vector is categorical or not
iscategorical(::CategoricalArray) = true
iscategorical(::AbstractVector{<: AbstractString}) = true
iscategorical(::AbstractVector{<: Symbol}) = true
iscategorical(::AbstractVector{<: Number}) = false
iscategorical(v::AbstractVector{<: Int}) = length(unique(v)) <= 6

################################################################################
#                                Basic recipes                                 #
################################################################################

########################################
#            Identity line             #
########################################

@recipe function f(::Type{Val{:identityline}}, x, y, z)

    label --> "Identity"

    seriestype := :straightline
    tmp = [minimum(x), maximum(x)]
    x := tmp
    y := tmp
end|

@recipe function f(::Type{Val{:title}}, str)

    showaxis := false
    grid := false

    seriestype = :annotations

    ()
end

################################################################################
#                               Convergence plot                               #
################################################################################

@userplot Convergence

@recipe function f(c::Convergence; params = Union{Symbol, String}[])

    @assert length(c.args) == 1
    @assert eltype(c.args) <: FittedPumasModel

    arg = c.args[1]

    names = _paramnames(arg)      # a wrapper function around some logic to make use of multiple dispatch

    str_params = string.(params)

    data  = _convergencedata(arg) # again, a wrapper function.

    inds = []

    !isempty(params) && (inds = findall(x -> !(x in str_params), names))

    finalnames = deleteat!(names, inds)

    finaldata = @view(data[(x -> !(x in inds)).(axes(data)[1]), :])

    layout --> good_layout(length(finalnames) + 1)  # good_layout simply overrides the Plots layouter for n <= 4

    legend := :none # clutters up the plot too much

    # also plot the observed function value from Optim
    @series begin
        title := "Objective function"
        subplot := length(names) + 1
        link := :x # link the x-axes, there should be no difference

        (_objectivefunctionvalues(arg)) # the observed values
    end

    # Iterate through three variables simultaneously:
    # - the index of the data (to assign to the subplot),
    # - the name of the parameter (as determined by `_paramnames`)
    # - the convergence data to plot as series vectors
    for (i, name, trace) in zip(eachindex(names), names, eachslice(data; dims = 2))
        @series begin
            title := name
            subplot := i
            link := :x
            # @show name i trace # debug here
            trace # return series data to be plotted.  We let the user specify the seriestype.
        end
    end

    primary := false # do not add a legend entry here, do other nice things that Plots does as well
end


################################################################################
#                          Covariate plots versus ηs                           #
################################################################################

@userplot EtaCov

@recipe function f(
            ec::EtaCov;
            cvs = [],
            continuoustype = :scatter,
            discretetype = :boxplot,
            catmap = NamedTuple()
        )

    @show typeof(ec)

    @assert length(ec.args) == 1
    @assert eltype(ec.args) <: Union{PumasModel, FittedPumasModel}
    @assert(isempty(catmap) || catmap isa NamedTuple)

    legend --> :none

    # extract the provided model
    res = ec.args[1]

    # easy API for use, why bother with nested data structures?
    df = DataFrame(inspect(res))

    # retrieve all covariate names.  Here, it's assumed they're the same for each Subject.
    allcovnames = res.data[1].covariates |> keys

    covnames = isempty(cvs) ? allcovnames : cvs

    # get the index number, to find the relevant `η` (eta)
    covindices = findfirst.((==).(covnames), Ref(allcovnames))

    etanames = [Symbol("ebe_" * string(i)) for i in covindices] # is this always true?

    # create a named tuple
    calculated_iterable = (;zip(covnames,iscategorical.(getindex.(Ref(df), !, covnames)))...)

    # merge the category map, such that it takes priority over the automatically calculated values.
    covtypes = merge(calculated_iterable, catmap)

    # use our good layout function
    layout --> good_layout(length(covindices))

    for (i, covname, etaname) in zip(eachindex(covnames), covnames, etanames)
        @series begin
            # can tweak this, or allow user to
            seriestype := covtypes[covname] ? :boxplot : :scatter
            subplot := i

            title := string(covname)
            ylabel := "η" # do we need this?

            (df[:, covname], df[:, etaname])
        end
    end

    primary := false

end
