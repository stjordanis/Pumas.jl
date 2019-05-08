using PuMaS.NCA, Test, CSV
using PuMaS

file = PuMaS.example_nmtran_data("nca_test_data/dapa_IV")
data = CSV.read(file)

timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"
data.route = "iv"
ncapop = @test_nowarn read_nca(data, id=:ID, time=:TIME, conc=:CObs, amt=:AMT_IV, route=:route,
                                    llq=0concu, timeu=timeu, concu=concu, amtu=amtu)
@test_nowarn NCA.auc(ncapop, method=:linuplogdown)
@test all(ismissing, NCA.bioav(ncapop, ithdose=1)[2])
@test_logs (:warn, "No dosage information has passed. If the dataset has dosage information, you can pass the column names by `amt=:AMT, route=:route`.") NCA.auc(read_nca(data, id=:ID, time=:TIME, conc=:CObs));
@test ncapop[1] isa NCASubject
@test ncapop[2:end-1] isa NCAPopulation

popncareport = NCAReport(ncapop, ithdose=1)
@test_nowarn popncareport
@test_broken display(NCA.to_markdown(popncareport))
@test_nowarn NCA.to_dataframe(popncareport)

lambdazdf = @test_nowarn NCA.lambdaz(ncapop)
@test size(lambdazdf, 2) == 2
@test lambdazdf[:lambdaz] isa Vector
@test lambdazdf[:id] == collect(1:24)
@test_nowarn NCA.lambdazr2(ncapop)
@test_nowarn NCA.lambdazadjr2(ncapop)
@test_nowarn NCA.lambdazintercept(ncapop)
@test_nowarn NCA.lambdaztimefirst(ncapop)
@test_nowarn NCA.lambdaznpoints(ncapop)

conc = Float64.(data[:CObs])*concu
t = Float64.(data[:TIME])*timeu
doses = Float64.(data[:AMT_IV])[1:16:end]*u"mg"

data = CSV.read(PuMaS.example_nmtran_data("nca_test_data/dapa_IV_sol"))

correct_auc = Float64.(data[:AUCINF_obs])*concu*timeu
correct_auc_last = Float64.(data[:AUClast])*concu*timeu
correct_aumc = Float64.(data[:AUMCINF_obs])./timeu

test_auc = rand(24,)
test_aumc = rand(24,)

idx = 1:16

nca = NCASubject(conc[idx], t[idx])
@test_logs (:warn, "`dose` is not provided. No dependent quantities will be calculated") NCAReport(nca)

@test (NCA.clast(zeros(5), 1:5), NCA.tlast(zeros(5), 1:5)) === (missing, missing)
@test (NCA.clast(1:5, 1:5), NCA.tlast(1:5, 1:5)) === (5, 5)
@test (NCA.clast(conc[idx], t[idx]), NCA.tlast(conc[idx], t[idx])) === (conc[idx][end], t[idx][end])
arr = [missing, 1, 2, 3, missing]
@test (NCA.clast(arr, 1:5), NCA.tlast(arr, 1:5)) === (3, 4)
@test (NCA.cmax(arr, 1:5), NCA.tmax(arr, 1:5)) === (3, 4)
@test (NCA.cmax(conc[idx], t[idx]), NCA.tmax(conc[idx], t[idx])) === (conc[idx][1], t[idx][1])
@test NCA.cmax(conc[idx], t[idx], interval=(2,Inf).*timeu) === conc[idx][7]
@test NCA.cmax(conc[idx], t[idx], interval=(24.,Inf).*timeu) === conc[idx][end]
@test NCA.cmax(conc[idx], t[idx], interval=[(24.,Inf).*timeu, (10.,Inf).*timeu]) == [NCA.cmax(nca, interval=(24.,Inf).*timeu), NCA.cmax(nca, interval=(10.,Inf).*timeu)]
@test NCA.tmax(conc[idx], t[idx], interval=[(24.,Inf).*timeu, (10.,Inf).*timeu]) == [NCA.tmax(nca, interval=(24.,Inf).*timeu), NCA.tmax(nca, interval=(10.,Inf).*timeu)]
@test_throws ArgumentError NCA.cmax(conc[idx], t[idx], interval=(100,Inf).*timeu)
x = (0:24.) .* timeu
ylg = NCA.interpextrapconc(conc[idx], t[idx], x; method=:linuplogdown)
yli = NCA.interpextrapconc(conc[idx], t[idx], x; method=:linear)
@test NCA.lambdaz(yli, collect(x), slopetimes=(10:20) .* timeu) ≈ NCA.lambdaz(ylg, collect(x), slopetimes=(10:20) .* timeu) atol=1e-2/timeu
@test NCA.auc(nca, interval=(t[end],Inf*timeu)) ≈ NCA.auc(nca) - NCA.auc(nca, auctype=:last) atol=1e-12*(concu*timeu)
c1 = ustrip(conc[2])
c2 = ustrip(conc[3])
t1 = ustrip(t[2])
t2 = ustrip(t[3])
@test NCA.c0(NCASubject(conc[2:5], t[2:5], dose=NCADose(0, 0.1, 0, NCA.IVBolus))) === exp(log(c1) - t1*(log(c2)-log(c1))/(t2-t1))*concu
@test NCA.c0(ncapop[1]) === ncapop[1].conc[1]
@test NCA.c0(nca) === missing
subj = NCASubject([10, 12.], [0.2, 0.9], dose=NCADose(0, 0.1, 0, NCA.IVBolus))
@test_logs (:warn, "c0: This is an IV bolus dose, but the first two concentrations are not decreasing. If `conc[i]/conc[i+1] > 0.8` holds, the back extrapolation will be computed internally for AUC and AUMC, but will not be reported.") NCA.c0(subj)
@test NCA.c0(subj, true, verbose=false) < 10

for m in (:linear, :linuplogdown, :linlog)
  @test_broken @inferred NCA.auc(conc[idx], t[idx], method=m)
  @test_broken @inferred NCA.aumc(conc[idx], t[idx], method=m)
  _nca = NCASubject(conc[idx], t[idx])
  @inferred NCA.auc( _nca, method=m)
  @inferred NCA.aumc(_nca, method=m)
  @test_nowarn NCA.interpextrapconc(conc[idx], t[idx], 1000rand(500)*timeu, method=m)
  @test_nowarn NCA.auc(conc[idx], t[idx], method=m, interval=(0,100.).*timeu, auctype=:last)
  @test_nowarn NCA.aumc(conc[idx], t[idx], method=m, interval=(0,100.).*timeu, auctype=:last)
  # test interval
  aucinf = NCA.auc(conc[idx], t[idx])
  aumcinf = NCA.aumc(conc[idx], t[idx])
  interval = [(0,Inf).*timeu, (0,Inf).*timeu, (0,Inf).*timeu]
  @test NCA.auc(conc[idx], t[idx], interval=interval) == [aucinf for i in 1:3]
  @test NCA.aumc(conc[idx], t[idx], interval=interval) == [aumcinf for i in 1:3]
  @test NCA.auc(conc[idx], t[idx], interval=(0,Inf).*timeu) === aucinf
  @test NCA.aumc(conc[idx], t[idx], interval=(0,Inf).*timeu) === aumcinf
  @test NCA.auc(conc[2:16], t[2:16].-t[2], method=m) == NCA.auc(conc[idx], t[idx], method=m, interval=(t[2], Inf*timeu))
  @test NCA.auc(conc[idx], t[idx], method=m, interval=(t[2], t[16])) ≈ NCA.auc(conc[2:16], t[2:16].-t[2], method=m, auctype=:last)

  auc10_in = NCA.auc(conc[idx], t[idx], interval=(0,23).*timeu) - NCA.auc(conc[idx], t[idx], interval=(10,23).*timeu)
  auc10_ex = NCA.auc(conc[idx], t[idx], interval=(0,50).*timeu) - NCA.auc(conc[idx], t[idx], interval=(10,50).*timeu)
  auc10  = NCA.auc(conc[idx], t[idx], interval=(0,10).*timeu)
  @test auc10_in ≈ auc10 ≈ auc10_ex
  aumc10_in = NCA.aumc(conc[idx], t[idx], interval=(0,23).*timeu) - NCA.aumc(conc[idx], t[idx], interval=(10,23).*timeu)
  aumc10_ex = NCA.aumc(conc[idx], t[idx], interval=(0,50).*timeu) - NCA.aumc(conc[idx], t[idx], interval=(10,50).*timeu)
  aumc10  = NCA.aumc(conc[idx], t[idx], interval=(0,10).*timeu)
  @test aumc10_in ≈ aumc10 ≈ aumc10_ex

  aucinf_ = NCA.auc(conc[idx], t[idx], interval=(0,21).*timeu) + NCA.auc(conc[idx], t[idx], interval=(21, Inf).*timeu)
  @test aucinf ≈ aucinf

  x = (0:.1:50) .* timeu
  y = NCA.interpextrapconc(conc[idx], t[idx], x; method=m)
  @test NCA.lambdaz(y, x) ≈ NCA.lambdaz(conc[idx], t[idx])

  @test_nowarn NCA.superposition(ncapop, 24timeu, method=m)
end

@test @inferred(NCA.lambdaz(nca, idxs=12:16)) == NCA.lambdaz(conc[idx], t[idx])
@test log(2)/NCA.lambdaz(conc[idx], t[idx]) === NCA.thalf(nca)
@test NCA.lambdaz(nca, slopetimes=t[10:13]) == NCA.lambdaz(conc[idx], t[idx], idxs=10:13)
@test NCA.lambdaz(nca, slopetimes=t[10:13]) !== NCA.lambdaz(conc[idx], t[idx])
@test NCA.lambdaz(nca, idxs=12:16) ≈ data[:Lambda_z][1]/timeu atol=1e-6/timeu

fails = (6,)
for i in 1:24
  idx = 16(i-1)+1:16*i
  dose = NCADose(0.0timeu, doses[i], nothing, NCA.IVInfusion)
  nca = ncapop[i]
  aucs = NCA.auc(nca, method=:linear)
  @test aucs === NCA.auc(conc[idx], t[idx], dose=dose, method=:linear)
  aumcs = NCA.aumc(nca, method=:linear)
  @test aumcs === NCA.aumc(conc[idx], t[idx], dose=dose, method=:linear)
  @test_nowarn NCA.auc(conc[idx], t[idx], method=:linuplogdown)
  @test_nowarn NCA.aumc(conc[idx], t[idx], method=:linuplogdown)
  aucps = NCA.auc(nca, method=:linear, pred=true)
  aumcps = NCA.aumc(nca, method=:linear, pred=true)
  if i in fails
    @test_broken data[:AUCINF_obs][i]*timeu*concu ≈ aucs atol = 1e-6*timeu*concu
    @test_broken data[:AUMCINF_obs][i]*timeu^2*concu ≈ aumcs atol = 1e-6*timeu^2*concu
    @test_broken data[:AUCINF_pred][i]*timeu*concu ≈ aucps atol = 1e-6*timeu*concu
    @test_broken data[:AUMCINF_pred][i]*timeu^2*concu ≈ aumcps atol = 1e-6*timeu^2*concu
    @test_broken data[:Lambda_z][i]/timeu ≈ NCA.lambdaz(nca) atol = 1e-6/timeu
    @test_broken data[:Lambda_z_intercept][i] ≈ NCA.lambdazintercept(nca) atol = 1e-6
    @test_broken data[:Clast_pred][i]*concu ≈ NCA.clast(nca, pred=true) atol = 1e-6*concu
    @test_broken data[:Vss_obs][i]*u"L" ≈ NCA.vss(nca) atol = 1e-6*u"L"
    @test_broken data[:Vz_obs][i]*u"L" ≈ NCA.vz(nca) atol = 1e-6*u"L"
  else
    @test data[:AUCINF_obs][i]*timeu*concu ≈ aucs atol = 1e-6*timeu*concu
    @test data[:AUMCINF_obs][i]*timeu^2*concu ≈ aumcs atol = 1e-6*timeu^2*concu
    @test data[:AUCINF_pred][i]*timeu*concu ≈ aucps atol = 1e-6*timeu*concu
    @test data[:AUMCINF_pred][i]*timeu^2*concu ≈ aumcps atol = 1e-6*timeu^2*concu
    @test data[:Lambda_z][i]/timeu ≈ NCA.lambdaz(nca) atol = 1e-6/timeu
    @test data[:Lambda_z_intercept][i] ≈ NCA.lambdazintercept(nca) atol = 1e-6
    @test data[:Clast_pred][i]*concu ≈ NCA.clast(nca, pred=true) atol = 1e-6*concu
    @test data[:Vss_obs][i]*u"L" ≈ NCA.vss(nca) atol = 1e-6*u"L"
    @test data[:Vz_obs][i]*u"L" ≈ NCA.vz(nca) atol = 1e-6*u"L"
  end
  aucs = NCA.auc(nca, dose=dose, method=:linear, auctype=:last)
  @test aucs == NCA.auclast(nca, dose=dose, method=:linear)
  aumcs = NCA.aumc(nca, dose=dose, method=:linear, auctype=:last)
  @test aumcs == NCA.aumclast(nca, dose=dose, method=:linear)
  @test normalizedose(aucs, nca) == aucs/doses[i]
  @test data[:AUClast][i]*timeu*concu ≈ aucs atol = 1e-6*timeu*concu
  @test data[:AUMClast][i]*timeu^2*concu ≈ aumcs atol = 1e-6*timeu^2*concu
  @test NCA.aumc_extrap_percent(nca) === NCA.aumc_extrap_percent(conc[idx], t[idx])
  @test NCA.auc_extrap_percent(nca) === NCA.auc_extrap_percent(conc[idx], t[idx])
  ncareport = @test_nowarn NCAReport(nca)
  i == 1 && @test_nowarn ncareport
  i == 1 && @test_nowarn display(NCA.to_markdown(ncareport))
  i == 1 && @test_nowarn NCA.to_dataframe(ncareport)
end

@test_nowarn NCA.c0(NCASubject([0.3, 0.2], [0.1, 0.2], dose=NCADose(0, 0.1, nothing, NCA.IVBolus)))
@test NCA.c0(NCASubject([0.3, 0.2], [0.1, 0.2], dose=NCADose(0, 0.1, nothing, NCA.EV))) === missing
@test NCA.c0(NCASubject([0.3, 0.2], [0.1, 0.2], dose=NCADose(0, 0.1, 1, NCA.IVInfusion))) === missing

df = DataFrame()
df.time = [0:20...; 20; 21:25]
df.conc = [0:20...; 0; 21:25]
df.amt = zeros(Int, 27); df.amt[22] = 1
df.route = "ev"
df.id = 1
@test_nowarn read_nca(df, llq=0concu, timeu=timeu, concu=concu, amtu=amtu)

df = DataFrame()
df.id=fill(1, 7); df.time=1:7; df.conc=[0, 0, 1, 1, 0, 1, 0]; df.blq=[1, 0, 0, 0, 1, 0, 0]
subj = read_nca(df, verbose=false)[1]
@test subj.time == findall(iszero, df.blq)
