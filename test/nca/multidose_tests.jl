using PuMaS.NCA, Test, CSV
using PuMaS

multiple_doses_file = PuMaS.example_nmtran_data("nca_test_data/dapa_IV_ORAL")
mdata = CSV.read(multiple_doses_file)
msol = CSV.read(PuMaS.example_nmtran_data("nca_test_data/dapa_IV_ORAL_sol"))

mncapop = @test_nowarn parse_ncadata(mdata, time=:TIME, conc=:COBS, amt=:AMT, formulation=:FORMULATION, occasion=:OCC)

@test_nowarn bioavailability(mncapop, 1)
@test all(vcat(tlag(mncapop)...) .=== float.(msol[:Tlag]))
@test_nowarn mrt(mncapop; auctype=:inf)
@test_nowarn mrt(mncapop; auctype=:last)
@test_nowarn mat(mncapop)
@test_nowarn cl(mncapop, 1)
@test tmin(mncapop[1])[1] == 20
@test cmin(mncapop[1])[end] == mncapop[1].conc[end][end]
