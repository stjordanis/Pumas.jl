using PuMaS.NCA, Test, CSV
using PuMaS

multiple_doses_file = PuMaS.example_nmtran_data("nca_test_data/dapa_IV_ORAL")
mdata = CSV.read(multiple_doses_file)
msol = CSV.read(PuMaS.example_nmtran_data("nca_test_data/dapa_IV_ORAL_sol"))

mncapop = @test_nowarn parse_ncadata(mdata, time=:TIME, conc=:COBS, amt=:AMT, formulation=:FORMULATION, occasion=:OCC, iv="IV")

lambdazdf = @test_nowarn NCA.lambdaz(mncapop)
@test size(lambdazdf, 2) == 3
@test lambdazdf[:lambdaz] isa Vector
@test lambdazdf[:occasion] == repeat(collect(1:4), 24)
@test lambdazdf[:id] == repeat(collect(1:24), inner=4)
@test_nowarn NCA.lambdazr2(mncapop)
@test_nowarn NCA.lambdazadjr2(mncapop)
@test_nowarn NCA.lambdazintercept(mncapop)
@test_nowarn NCA.lambdaztimefirst(mncapop)
@test_nowarn NCA.lambdaznpoints(mncapop)

@test_nowarn NCA.bioav(mncapop, ithdose=1)
@test all(vcat(NCA.tlag(mncapop)[:tlag]...) .=== float.(msol[:Tlag]))
@test_nowarn NCA.mrt(mncapop; auctype=:inf)
@test_nowarn NCA.mrt(mncapop; auctype=:last)
@test_nowarn NCA.mat(mncapop)
@test_nowarn NCA.cl(mncapop, ithdose=1)
@test NCA.tmin(mncapop[1])[1] == 20
@test NCA.cmin(mncapop[1])[end] == mncapop[1].conc[end][end]
@test NCA.fluctation(mncapop[1]) == 100 .*(NCA.cmax(mncapop[1]) .- NCA.cmin(mncapop[1]))./NCA.cavg(mncapop[1])
@test NCA.accumulationindex(mncapop[1]) == inv.(1 .-exp.(-NCA.lambdaz(mncapop[1]).*NCA.tau(mncapop[1])))
@test NCA.swing(mncapop[1]) == (NCA.cmax(mncapop[1]) .- NCA.cmin(mncapop[1]))./NCA.cmin(mncapop[1])
@test NCA.c0(mncapop[1]) == NCA.cmin(mncapop[1])
@test NCA.c0(mncapop[1], method=:set0) == zero(NCA.cmin(mncapop[1]))

@test_nowarn display(NCAReport(mncapop[1], ithdose=1))
