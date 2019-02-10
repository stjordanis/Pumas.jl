using PuMaS.NCA, Test, CSV
using PuMaS

multiple_doses_file = PuMaS.example_nmtran_data("nca_test_data/dapa_IV_ORAL")
mdata = CSV.read(multiple_doses_file)
msol = CSV.read(PuMaS.example_nmtran_data("nca_test_data/dapa_IV_ORAL_sol"))

mncapop = @test_nowarn parse_ncadata(mdata, time=:TIME, conc=:COBS, amt=:AMT, formulation=:FORMULATION, occasion=:OCC, iv="IV")

@test_nowarn NCA.bioav(mncapop, 1)
@test all(vcat(NCA.tlag(mncapop)[2]...) .=== float.(msol[:Tlag]))
@test_nowarn NCA.mrt(mncapop; auctype=:inf)
@test_nowarn NCA.mrt(mncapop; auctype=:last)
@test_nowarn NCA.mat(mncapop)
@test_nowarn NCA.cl(mncapop, 1)
@test NCA.tmin(mncapop[1])[1] == 20
@test NCA.cmin(mncapop[1])[end] == mncapop[1].conc[end][end]
@test NCA.fluctation(mncapop[1]) == 100 .*(NCA.cmax(mncapop[1]) .- NCA.cmin(mncapop[1]))./NCA.cavg(mncapop[1])
@test NCA.accumulationindex(mncapop[1]) == inv.(1 .-exp.(-NCA.lambdaz(mncapop[1]).*NCA.tau(mncapop[1])))
@test NCA.swing(mncapop[1]) == (NCA.cmax(mncapop[1]) .- NCA.cmin(mncapop[1]))./NCA.cmin(mncapop[1])
@test NCA.c0(mncapop[1]) == NCA.cmin(mncapop[1])
@test NCA.c0(mncapop[1], method=:set0) == zero(NCA.cmin(mncapop[1]))
