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

@recipe function f(::Type{Val{:title}}, str)

    showaxis := false
    grid := false

    seriestype = :annotations

    ()
end
