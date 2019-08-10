################################################################################
#                                Basic recipes                                 #
################################################################################

########################################
#                Loess                 #
########################################

@recipe function f(::Type{Val{:loess}}, x, y, z)
    ox, oy = x, y

    loess_resolution = minimum(diff(ox)) / 4 # the minimum diffference in xs, div by 4.  TODO is there a better heuristic?

    model = Loess.loess(ox, oy)
    loess_fit_xs = range(minimum(ox), maximum(ox), step=loess_resolution)
    loess_fit_ys = Loess.predict(model, loess_fit_xs)

    label --> "LOESS fit"

    seriestype := :path
    x := loess_fit_xs
    y := loess_fit_ys
end

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

@recipe function f(::Type{Val{:title}}, str)

    showaxis := false
    grid := false

    seriestype = :annotations

    ()
end
