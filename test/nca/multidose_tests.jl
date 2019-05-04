using PuMaS.NCA, Test, CSV
using PuMaS

multiple_doses_file = PuMaS.example_nmtran_data("nca_test_data/dapa_IV_ORAL")
mdata = CSV.read(multiple_doses_file)
msol = CSV.read(PuMaS.example_nmtran_data("nca_test_data/dapa_IV_ORAL_sol"))

timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"
mdata.route = map(f -> f=="ORAL" ? "ev" : "iv", mdata.FORMULATION)
mncapop = @test_nowarn read_nca(mdata, time=:TIME, conc=:COBS, amt=:AMT, route=:route, occasion=:OCC,
                                     timeu=timeu, concu=concu, amtu=amtu)

@test_throws ArgumentError NCA.interpextrapconc(mncapop[1], 22timeu, method=:linear)

# test caching
@test_nowarn NCA.auc(mncapop)
@test NCA.auc(mncapop[1]; method=:linear) != NCA.auc(mncapop[1]; method=:linlog)
@test NCA.auc(mncapop[1]; method=:linear) != NCA.auc(mncapop[1]; method=:linuplogdown)

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
@test all(vcat(NCA.tlag(mncapop)[:tlag]...) .=== float.(map(x -> ismissing(x) ? x : x*timeu, msol[:Tlag])))
@test_nowarn NCA.mrt(mncapop; auctype=:inf)
@test_nowarn NCA.mrt(mncapop; auctype=:last)
@test_nowarn NCA.mat(mncapop)
@test_nowarn NCA.cl(mncapop, ithdose=1)
@test NCA.tmin(mncapop[1])[1] == 20timeu
@test NCA.cmin(mncapop[1])[end] == mncapop[1].conc[end][end]
@test NCA.fluctation(mncapop[1]) == 100 .*(NCA.cmax(mncapop[1]) .- NCA.cmin(mncapop[1]))./NCA.cavg(mncapop[1])
@test NCA.accumulationindex(mncapop[1]) == inv.(1 .-exp.(-NCA.lambdaz(mncapop[1]).*NCA.tau(mncapop[1])))
@test NCA.swing(mncapop[1]) == (NCA.cmax(mncapop[1]) .- NCA.cmin(mncapop[1]))./NCA.cmin(mncapop[1])
@test NCA.c0(mncapop[1])[1] == mncapop[1].conc[1][1]

ncareport1 = NCAReport(mncapop[1], ithdose=1)
@test_nowarn ncareport1
@test_nowarn display(NCA.to_markdown(ncareport1))
@test_nowarn NCA.to_dataframe(ncareport1)

popncareport = NCAReport(mncapop, ithdose=1)
@test_nowarn popncareport
@test_broken display(NCA.to_markdown(popncareport))
@test_nowarn NCA.to_dataframe(popncareport)

# test ii
# scalar
ii1 = 0.1timeu
pop = @test_nowarn read_nca(mdata, time=:TIME, conc=:COBS, amt=:AMT, occasion=:OCC, route=:route, timeu=timeu, concu=concu, amtu=amtu, ii=ii1)
@test all(subj -> subj.ii == ii1, pop)
@test NCA.tau(pop[1]) == [ii1 for i in 1:4]
@test NCA.accumulationindex(pop[1]) == inv.(1 .-exp.(-NCA.lambdaz(mncapop[1]).*ii1))
# vector
ii2 = [i*timeu for i in eachindex(pop)]
pop = @test_nowarn read_nca(mdata, time=:TIME, conc=:COBS, amt=:AMT, occasion=:OCC, route=:route, timeu=timeu, concu=concu, amtu=amtu, ii=ii2)
@test all(i -> pop[i].ii == ii2[i], eachindex(pop))
# length error
ii3 = [i*timeu for i in 1:100]
@test_throws ArgumentError read_nca(mdata, time=:TIME, conc=:COBS, amt=:AMT, occasion=:OCC, route=:route, timeu=timeu, concu=concu, amtu=amtu, ii=ii3)
# type error
@test_throws Unitful.DimensionError read_nca(mdata, time=:TIME, conc=:COBS, amt=:AMT, occasion=:OCC, route=:route, timeu=timeu, concu=concu, amtu=amtu, ii=2)
@test_throws ArgumentError read_nca(mdata, time=:TIME, conc=:COBS, amt=:AMT, occasion=:OCC, route=:route, timeu=timeu, concu=concu, amtu=amtu, ii=:ID)

data1 = CSV.read(IOBuffer("""
id,time,tad,conc,amt,occasion,formulation
1,0.0,0,0.755,0.705,1,oral
1,1.0,1,0.55,0,1,oral
1,2.0,2,0.65,0,1,oral
1,12.0,0,0.473,0.29,2,oral
1,13.0,1,0.235,0,2,oral
1,14.0,2,0.109,0,2,oral
2,0.0,0,0.341,0.932,1,oral
2,1.0,1,0.557,0,1,oral
2,2.0,2,0.159,0,1,oral
2,3.0,0,0.307,0,2,oral
2,3.5,0,0.226,0,3,oral
3,0.0,0,0.763,0.321,1,oral
3,1.0,1,0.96,0,1,oral
3,2.0,2,0.772,0,1,oral
3,12.0,0,0.941,0.656,2,oral
3,13.0,1,0.204,0,2,oral
3,14.0,2,0.0302,0,2,oral
"""))
data2 = CSV.read(IOBuffer("""
id,time,tad,conc,amt,occasion,formulation
1,0.0,0,0.755,0.705,1,oral
1,1.0,1,0.55,0,1,oral
1,2.0,2,0.65,0,1,oral
1,12.0,0,0.473,0.29,2,oral
1,13.0,1,0.235,0,2,oral
1,14.0,2,0.109,0,2,oral
2,0.0,0,0.341,0.932,1,oral
2,1.0,1,0.557,0,1,oral
2,2.0,2,0.159,0,1,oral
3,0.0,0,0.763,0.321,1,oral
3,1.0,1,0.96,0,1,oral
3,2.0,2,0.772,0,1,oral
3,12.0,0,0.941,0.656,2,oral
3,13.0,1,0.204,0,2,oral
3,14.0,2,0.0302,0,2,oral
"""))
for df in (data1, data2)
  df.route = "ev"
  @test_throws AssertionError read_nca(df, id=:id,
                      route=:route,
                      occasion=:occasion,
                      amt=:amt,
                      timeu=timeu, concu=concu, amtu=amtu);
end
