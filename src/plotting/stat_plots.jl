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

################################################################################
#                       Covariate plots versus residuals                       #
################################################################################

@userplot ResPlot

@recipe function f(
            rp::ResPlot;
            cvs = [],
            continuoustype = :scatter,
            discretetype = :boxplot,
            catmap = NamedTuple()
        )

    @show typeof(ec)

    @assert length(rp.args) == 1
    @assert eltype(rp.args) <: Union{PumasModel, FittedPumasModel}
    @assert(isempty(catmap) || catmap isa NamedTuple)

    legend --> :none

    res = ec.args[1]

    df = DataFrame(inspect(res))

    ws = wresiduals(res)

    allcovnames = res.data[1].covariates |> keys

    covnames = isempty(cvs) ? allcovnames : cvs # here, it's assumed covariates are the same for all Subjects.

    covindices = findfirst.((==).(covnames), Ref(allcovnames)) # get the index number, to find the relevant `η` (eta)

    etanames = [Symbol("ebe_" * string(i)) for i in covindices] # is this always true?

    calculated_iterable = (;zip(covnames,iscategorical.(getindex.(Ref(df), !, covnames)))...)

    covtypes = merge(calculated_iterable, catmap)

    layout --> good_layout(length(covindices))

    for (i, covname, etaname) in zip(eachindex(covnames), covnames, etanames)
        @series begin
            seriestype := covtypes[covname] ? :boxplot : :scatter
            subplot := i

            title := string(covname)
            ylabel := "η"

            (df[:, covname], df[:, etaname])
        end
    end

    primary := false

end
