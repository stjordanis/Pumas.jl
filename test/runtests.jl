using PuMaS, SafeTestsets

if haskey(ENV,"GROUP")
    group = ENV["GROUP"]
else
    group = "All"
end

is_APPVEYOR = ( Sys.iswindows() && haskey(ENV,"APPVEYOR") )


@time begin

if group == "All" || group == "Core"
    include("core/runtests.jl")
end

if group == "All" || group == "NCA"
    include("nca/runtests.jl")
end

if group == "All" || group == "NLME_ML1" || group == "NLME_ML2" ||group == "NLME_BAYES"
    include("nlme/runtests.jl")
end

if group == "All" || group == "Features"
    include("features/runtests.jl")
end



end
