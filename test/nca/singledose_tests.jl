using PuMaS.NCA, Test, CSV
using PuMaS
using Random

file = PuMaS.example_nmtran_data("nca_test_data/dapa_IV")
data = CSV.read(file)

ncapop = @test_nowarn parse_ncadata(data, time=:TIME, conc=:CObs, amt=:AMT_IV, formulation=:Formulation, iv="IV")
@test_nowarn NCA.auc(ncapop)
@test all(ismissing, NCA.bioav(ncapop, 1)[2])
@test_logs (:warn, "No dosage information has passed. If the dataset has dosage information, you can pass the column names by\n    `amt=:AMT, formulation=:FORMULATION, iv=\"IV\"`") NCA.auc(parse_ncadata(data, time=:TIME, conc=:CObs));
@test ncapop[1] isa NCASubject
@test ncapop[2:end-1] isa NCAPopulation

@test_nowarn NCA.lambdaz(ncapop)
@test_nowarn NCA.lambdazr2(ncapop)
@test_nowarn NCA.lambdazadjr2(ncapop)
@test_nowarn NCA.lambdazintercept(ncapop)
@test_nowarn NCA.lambdaztimefirst(ncapop)

conc = Float64.(data[:CObs])
t = Float64.(data[:TIME])
doses = Float64.(data[:AMT_IV])[1:16:end]

data = CSV.read(PuMaS.example_nmtran_data("nca_test_data/dapa_IV_sol"))

correct_auc = Float64.(data[:AUCINF_obs])
correct_auc_last = Float64.(data[:AUClast])
correct_aumc = Float64.(data[:AUMCINF_obs])

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
@test NCA.cmax(conc[idx], t[idx], interval=(2,Inf)) === conc[idx][7]
@test NCA.cmax(conc[idx], t[idx], interval=(24.,Inf)) === conc[idx][end]
@test NCA.cmax(conc[idx], t[idx], interval=[(24.,Inf), (10.,Inf)]) == [NCA.cmax(nca, interval=(24.,Inf)), NCA.cmax(nca, interval=(10.,Inf))]
@test NCA.tmax(conc[idx], t[idx], interval=[(24.,Inf), (10.,Inf)]) == [NCA.tmax(nca, interval=(24.,Inf)), NCA.tmax(nca, interval=(10.,Inf))]
@test_throws ArgumentError NCA.cmax(conc[idx], t[idx], interval=(100,Inf))
x = 0:24.
ylg = NCA.interpextrapconc(conc[idx], t[idx], x; interpmethod=:linuplogdown)
yli = NCA.interpextrapconc(conc[idx], t[idx], x; interpmethod=:linear)
@test NCA.lambdaz(yli, collect(x), slopetimes=10:20) ≈ NCA.lambdaz(ylg, collect(x), slopetimes=10:20) atol=1e-2
@test NCA.auc(nca, interval=(t[end],Inf)) ≈ NCA.auc(nca) - NCA.auc(nca, auctype=:last) atol=1e-12
@test NCA.c0(ncapop[1]) == ncapop[1].conc[1]
@test NCA.c0(nca, method=:set0) == zero(nca.conc[1])
@test NCA.c0(ncapop[1], method=:c0) === ncapop[1].conc[1]
@test NCA.c0(ncapop[1], method=:c1) === ncapop[1].conc[2]
@test_nowarn NCA.c0(ncapop[1], method=:logslope)

for m in (:linear, :linuplogdown, :linlog)
  @test_broken @inferred NCA.auc(conc[idx], t[idx], method=m)
  @test_broken @inferred NCA.aumc(conc[idx], t[idx], method=m)
  _nca = NCASubject(conc[idx], t[idx])
  @inferred NCA.auc( _nca, method=m)
  @inferred NCA.aumc(_nca, method=m)
  @test_nowarn NCA.interpextrapconc(conc[idx], t[idx], 1000rand(500), interpmethod=m)
  @test_nowarn NCA.auc(conc[idx], t[idx], method=m, interval=(0,100.), auctype=:last)
  @test_nowarn NCA.aumc(conc[idx], t[idx], method=m, interval=(0,100.), auctype=:last)
  # test interval
  aucinf = NCA.auc(conc[idx], t[idx])
  aumcinf = NCA.aumc(conc[idx], t[idx])
  @test NCA.auc(conc[idx], t[idx], interval=[(0,Inf), (0,Inf), (0,Inf)]) == [aucinf for i in 1:3]
  @test NCA.aumc(conc[idx], t[idx], interval=[(0,Inf), (0,Inf), (0,Inf)]) == [aumcinf for i in 1:3]
  @test NCA.auc(conc[idx], t[idx], interval=(0,Inf)) === aucinf
  @test NCA.aumc(conc[idx], t[idx], interval=(0,Inf)) === aumcinf
  @test NCA.auc(conc[2:16], t[2:16], method=m) == NCA.auc(conc[idx], t[idx], method=m, interval=(t[2], Inf))
  @test NCA.auc(conc[2:16], t[2:16], method=m, interval=(t[2], Inf), auctype=:last) == NCA.auc(conc[2:16], t[2:16], method=m, interval=(t[2], t[16]), auctype=:last)

  auc10_in = NCA.auc(conc[idx], t[idx], interval=(0,23)) - NCA.auc(conc[idx], t[idx], interval=(10,23))
  auc10_ex = NCA.auc(conc[idx], t[idx], interval=(0,50)) - NCA.auc(conc[idx], t[idx], interval=(10,50))
  auc10  = NCA.auc(conc[idx], t[idx], interval=(0,10))
  @test auc10_in ≈ auc10 ≈ auc10_ex
  aumc10_in = NCA.aumc(conc[idx], t[idx], interval=(0,23)) - NCA.aumc(conc[idx], t[idx], interval=(10,23))
  aumc10_ex = NCA.aumc(conc[idx], t[idx], interval=(0,50)) - NCA.aumc(conc[idx], t[idx], interval=(10,50))
  aumc10  = NCA.aumc(conc[idx], t[idx], interval=(0,10))
  @test aumc10_in ≈ aumc10 ≈ aumc10_ex

  aucinf_ = NCA.auc(conc[idx], t[idx], interval=(0,21)) + NCA.auc(conc[idx], t[idx], interval=(21, Inf))
  @test aucinf ≈ aucinf

  x = 0:.1:50
  y = NCA.interpextrapconc(conc[idx], t[idx], x; interpmethod=m)
  @test NCA.lambdaz(y, x) ≈ NCA.lambdaz(conc[idx], t[idx])
end

@test @inferred(NCA.lambdaz(nca, idxs=12:16)) == NCA.lambdaz(conc[idx], t[idx])
@test log(2)/NCA.lambdaz(conc[idx], t[idx]) === NCA.thalf(nca)
@test NCA.lambdaz(nca, slopetimes=t[10:13]) == NCA.lambdaz(conc[idx], t[idx], idxs=10:13)
@test NCA.lambdaz(nca, slopetimes=t[10:13]) !== NCA.lambdaz(conc[idx], t[idx])
@test NCA.lambdaz(nca, idxs=12:16) ≈ data[:Lambda_z][1] atol=1e-6

fails = (6,)
for i in 1:24
  idx = 16(i-1)+1:16*i
  dose = NCADose(0.0, doses[i], NCA.IV)
  nca = NCASubject(conc[idx], t[idx], dose=dose)
  aucs = NCA.auc(nca, method=:linear)
  @test aucs === NCA.auc(conc[idx], t[idx], dose=dose, method=:linear)
  aumcs = NCA.aumc(nca, method=:linear)
  @test aumcs === NCA.aumc(conc[idx], t[idx], dose=dose, method=:linear)
  @test_nowarn NCA.auc(conc[idx], t[idx], method=:linuplogdown)
  @test_nowarn NCA.aumc(conc[idx], t[idx], method=:linuplogdown)
  aucps = NCA.auc(nca, method=:linear, pred=true)
  aumcps = NCA.aumc(nca, method=:linear, pred=true)
  if i in fails
    @test_broken data[:AUCINF_obs][i] ≈ aucs atol = 1e-6
    @test_broken data[:AUMCINF_obs][i] ≈ aumcs atol = 1e-6
    @test_broken data[:AUCINF_pred][i] ≈ aucps atol = 1e-6
    @test_broken data[:AUMCINF_pred][i] ≈ aumcps atol = 1e-6
    @test_broken data[:Lambda_z][i] ≈ NCA.lambdaz(nca) atol = 1e-6
    @test_broken data[:Lambda_z_intercept][i] ≈ NCA.lambdazintercept(nca) atol = 1e-6
    @test_broken data[:Clast_pred][i] ≈ NCA.clast(nca, pred=true) atol = 1e-6
    @test_broken data[:Vss_obs][i] ≈ NCA.vss(nca) atol = 1e-6
    @test_broken data[:Vz_obs][i] ≈ NCA.vz(nca) atol = 1e-6
  else
    @test data[:AUCINF_obs][i] ≈ aucs atol = 1e-6
    @test data[:AUMCINF_obs][i] ≈ aumcs atol = 1e-6
    @test data[:AUCINF_pred][i] ≈ aucps atol = 1e-6
    @test data[:AUMCINF_pred][i] ≈ aumcps atol = 1e-6
    @test data[:Lambda_z][i] ≈ NCA.lambdaz(nca) atol = 1e-6
    @test data[:Lambda_z_intercept][i] ≈ NCA.lambdazintercept(nca) atol = 1e-6
    @test data[:Clast_pred][i] ≈ NCA.clast(nca, pred=true) atol = 1e-6
    @test data[:Vss_obs][i] ≈ NCA.vss(nca) atol = 1e-6
    @test data[:Vz_obs][i] ≈ NCA.vz(nca) atol = 1e-6
  end
  aucs = NCA.auc(nca, dose=dose, method=:linear, auctype=:last)
  aumcs = NCA.aumc(nca, dose=dose, method=:linear, auctype=:last)
  @test normalizedose(aucs, nca) == aucs/doses[i]
  @test data[:AUClast][i] ≈ aucs atol = 1e-6
  @test data[:AUMClast][i] ≈ aumcs atol = 1e-6
  @test NCA.aumc_extrap_percent(nca) === NCA.aumc_extrap_percent(conc[idx], t[idx])
  @test NCA.auc_extrap_percent(nca) === NCA.auc_extrap_percent(conc[idx], t[idx])
  ncareport = @test_nowarn NCAReport(nca)
  i == 1 && @test_nowarn display(ncareport)
end
