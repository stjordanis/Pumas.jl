################################################################################
#                           Useful utility functions                           #
################################################################################

# heuristic to determine if a vector is categorical or not
"""
    iscategorical(arr)
Returns whether that array can be considered categorical or not, according to
the following heuristic:

- A CategoricalArray is always categorical.
- A `Vector` of `Strings` or `Symbols` is always categorical.
- A `Vector` of integers, with no. of unique entries ≤ 6, is always categorical.
- A `Vector` of `Numbers` is never categorical, it is always continuous.
"""
iscategorical(::CategoricalArray) = true
iscategorical(::AbstractVector{<: AbstractString}) = true
iscategorical(::AbstractVector{<: Symbol}) = true
iscategorical(::AbstractVector{<: Number}) = false
iscategorical(::AbstractVector{Union{Missing, T}}) where T = false
# iscategorical(v::AbstractVector{<: Int}) = length(unique(v)) <= 6 # TODO is this heuristic OK?

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
end

@recipe function f(::Type{Val{:title}}, str, y, z)

    showaxis := false
    grid := false

    seriestype = :annotations

    ()
end

################################################################################
#                  Advanced type recipes and specializations                   #
################################################################################

# You can check the `plotattributes` dict, which is accessible in any recipe,
# for any plot attribute.

# @recipe function f()

################################################################################
#                               Convergence plot                               #
################################################################################

"""
    convergence(res::FittedPumasModel; params = [])

Plots the convergence of the FittedPumasModel `res`, by plotting the values taken
by individual parameters against the iteration.

The `params` keyword should be passed a `Vector` of symbols, which represent the
parameters to be plotted.
"""
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
