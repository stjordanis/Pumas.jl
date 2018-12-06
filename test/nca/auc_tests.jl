using PuMaS.NCA, Test, CSV
using PuMaS
using Random

file = PuMaS.example_nmtran_data("nca_test_data/dapa_IV")
data = CSV.read(file)

ncapop = @test_nowarn parse_ncadata(data, time=:TIME, conc=:CObs, amt=:AMT_IV, formulation=:Formulation)
@test_nowarn auc(ncapop)

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

nca = NCAData(conc[idx], t[idx])
@test_logs (:warn, "`dose` is not provided. No dependent quantities will be calculated") NCAReport(nca)

@test (clast(zeros(5), 1:5), tlast(zeros(5), 1:5)) === (missing, missing)
@test (clast(1:5, 1:5), tlast(1:5, 1:5)) === (5, 5)
@test (clast(conc[idx], t[idx]), tlast(conc[idx], t[idx])) === (conc[idx][end], t[idx][end])
arr = [missing, 1, 2, 3, missing]
@test (clast(arr, 1:5), tlast(arr, 1:5)) === (3, 4)
@test (cmax(arr, 1:5), tmax(arr, 1:5)) === (3, 4)
@test (cmax(conc[idx], t[idx]), tmax(conc[idx], t[idx])) === (conc[idx][1], t[idx][1])
@test cmax(conc[idx], t[idx], interval=(2,Inf)) === conc[idx][7]
@test cmax(conc[idx], t[idx], interval=(24.,Inf)) === conc[idx][end]
@test cmax(conc[idx], t[idx], interval=(24.,Inf), dose=NCADose(1, NCA.IV, 2))[2] === conc[idx][end]/2
@test cmax(conc[idx], t[idx], interval=[(24.,Inf), (10.,Inf)]) == [cmax(nca, interval=(24.,Inf)), cmax(nca, interval=(10.,Inf))]
@test tmax(conc[idx], t[idx], interval=[(24.,Inf), (10.,Inf)]) == [tmax(nca, interval=(24.,Inf)), tmax(nca, interval=(10.,Inf))]
@test_throws ArgumentError cmax(conc[idx], t[idx], interval=(100,Inf))
x = 0:24.
ylg = NCA.interpextrapconc(conc[idx], t[idx], x; interpmethod=:linuplogdown)
yli = NCA.interpextrapconc(conc[idx], t[idx], x; interpmethod=:linear)
@test lambdaz(yli, collect(x), slopetimes=10:20)[1] ≈ lambdaz(ylg, collect(x), slopetimes=10:20)[1] atol=1e-2

for m in (:linear, :linuplogdown, :linlog)
  @inferred auc(conc[idx], t[idx], method=m)
  @inferred aumc(conc[idx], t[idx], method=m)
  @test_nowarn NCA.interpextrapconc(conc[idx], t[idx], 1000rand(500), interpmethod=m)
  @test_nowarn auc(conc[idx], t[idx], method=m, interval=(0,100.), auctype=:AUClast)
  @test_nowarn aumc(conc[idx], t[idx], method=m, interval=(0,100.), auctype=:AUMClast)
  # test interval
  aucinf = auc(conc[idx], t[idx])
  aumcinf = aumc(conc[idx], t[idx])
  @test auc(conc[idx], t[idx], interval=[(0,Inf), (0,Inf), (0,Inf)]) == [aucinf for i in 1:3]
  @test aumc(conc[idx], t[idx], interval=[(0,Inf), (0,Inf), (0,Inf)]) == [aumcinf for i in 1:3]
  @test auc(conc[idx], t[idx], interval=(0,Inf)) === aucinf
  @test aumc(conc[idx], t[idx], interval=(0,Inf)) === aumcinf
  @test auc(conc[2:16], t[2:16], method=m) == auc(conc[idx], t[idx], method=m, interval=(t[2], Inf))
  @test auc(conc[2:16], t[2:16], method=m, interval=(t[2], Inf), auctype=:AUClast) == auc(conc[2:16], t[2:16], method=m, interval=(t[2], t[16]), auctype=:AUClast)

  auc10_in = auc(conc[idx], t[idx], interval=(0,23)) - auc(conc[idx], t[idx], interval=(10,23))
  auc10_ex = auc(conc[idx], t[idx], interval=(0,50)) - auc(conc[idx], t[idx], interval=(10,50))
  auc10  = auc(conc[idx], t[idx], interval=(0,10))
  @test auc10_in ≈ auc10 ≈ auc10_ex
  aumc10_in = aumc(conc[idx], t[idx], interval=(0,23)) - aumc(conc[idx], t[idx], interval=(10,23))
  aumc10_ex = aumc(conc[idx], t[idx], interval=(0,50)) - aumc(conc[idx], t[idx], interval=(10,50))
  aumc10  = aumc(conc[idx], t[idx], interval=(0,10))
  @test aumc10_in ≈ aumc10 ≈ aumc10_ex

  aucinf_ = auc(conc[idx], t[idx], interval=(0,21)) + auc(conc[idx], t[idx], interval=(21, Inf))
  @test aucinf ≈ aucinf

  x = 0:.1:50
  y = NCA.interpextrapconc(conc[idx], t[idx], x; interpmethod=m)
  @test lambdaz(y, x)[1] ≈ lambdaz(conc[idx], t[idx])[1]
end

@test @inferred(lambdaz(nca, idxs=12:16))[1] == lambdaz(conc[idx], t[idx])[1]
@test log(2)/lambdaz(conc[idx], t[idx])[1] === thalf(nca)
@test lambdaz(nca, slopetimes=t[10:13])[1] == lambdaz(conc[idx], t[idx], idxs=10:13)[1]
@test lambdaz(nca, slopetimes=t[10:13])[1] !== lambdaz(conc[idx], t[idx])[1]
@test lambdaz(nca, idxs=12:16)[1] ≈ data[:Lambda_z][1] atol=1e-6

fails = (6, 9)
for i in 1:24
  idx = 16(i-1)+1:16*i
  dose = NCADose(1, NCA.IV, doses[i])
  nca = NCAData(conc[idx], t[idx], dose=dose)
  aucs = auc(nca, method=:linear)
  @test aucs === auc(conc[idx], t[idx], dose=dose, method=:linear)
  aumcs = aumc(nca, method=:linear)
  @test aumcs === aumc(conc[idx], t[idx], dose=dose, method=:linear)
  @test_nowarn auc(conc[idx], t[idx], method=:linuplogdown)
  @test_nowarn aumc(conc[idx], t[idx], method=:linuplogdown)
  @test aucs[2] == aucs[1]/doses[i]
  @test aumcs[2] == aumcs[1]/doses[i]
  if i in fails
    @test_broken data[:AUCINF_obs][i] ≈ aucs[1] atol = 1e-6
    @test_broken data[:AUMCINF_obs][i] ≈ aumcs[1] atol = 1e-6
    @test_broken data[:Lambda_z][i] ≈ lambdaz(nca)[1] atol = 1e-6
    @test_broken data[:Vss_obs][i] ≈ vss(nca) atol = 1e-6
    @test_broken data[:Vz_obs][i] ≈ vz(nca) atol = 1e-6
  else
    @test data[:AUCINF_obs][i] ≈ aucs[1] atol = 1e-6
    @test data[:AUMCINF_obs][i] ≈ aumcs[1] atol = 1e-6
    @test data[:Lambda_z][i] ≈ lambdaz(nca)[1] atol = 1e-6
    @test data[:Vss_obs][i] ≈ vss(nca) atol = 1e-6
    @test data[:Vz_obs][i] ≈ vz(nca) atol = 1e-6
  end
  aucs = auc(nca, dose=dose, method=:linear, auctype=:AUClast)
  aumcs = aumc(nca, dose=dose, method=:linear, auctype=:AUMClast)
  @test aucs[2] == aucs[1]/doses[i]
  @test aumcs[2] == aumcs[1]/doses[i]
  @test data[:AUClast][i] ≈ aucs[1] atol = 1e-6
  @test data[:AUMClast][i] ≈ aumcs[1] atol = 1e-6
  @test aumc_extrap_percent(nca) === aumc_extrap_percent(conc[idx], t[idx])
  @test auc_extrap_percent(nca) === auc_extrap_percent(conc[idx], t[idx])
  @test_nowarn NCAReport(nca)
end
