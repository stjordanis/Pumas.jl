using PuMaS.NCA, Test, CSV
using PuMaS
using Random

root = joinpath(dirname(pathof(PuMaS)), "..")
data = CSV.read("$root/examples/nca_test_data/dapa_IV.csv")

conc = Float64.(data[:CObs])
t = Float64.(data[:TIME])
dose = Float64.(data[:AMT_IV])[1:16:end]

data = CSV.read("$root/examples/nca_test_data/Final_Parameters_Pivoted.csv")

correct_auc = Float64.(data[:AUCINF_obs])
correct_auc_last = Float64.(data[:AUClast])
correct_aumc = Float64.(data[:AUMCINF_obs])

test_auc = rand(24,)
test_aumc = rand(24,)

idx = 1:16

@test (clast(zeros(5), 1:5), tlast(zeros(5), 1:5)) === (missing, missing)
@test (clast(1:5, 1:5), tlast(1:5, 1:5)) === (5, 5)
@test (clast(conc[idx], t[idx]), tlast(conc[idx], t[idx])) === (conc[idx][end], t[idx][end])
arr = [missing, 1, 2, 3, missing]
@test (clast(arr, 1:5), tlast(arr, 1:5)) === (3, 4)
@test (cmax(arr, 1:5), tmax(arr, 1:5)) === (3, 4)
@test (cmax(conc[idx], t[idx]), tmax(conc[idx], t[idx])) === (conc[idx][1], t[idx][1])
@test cmax(conc[idx], t[idx], interval=(2,Inf)) === conc[idx][7]
@test cmax(conc[idx], t[idx], interval=(24.,Inf)) === conc[idx][end]
@test_throws ArgumentError cmax(conc[idx], t[idx], interval=(100,Inf))
x = 0:24.
ylg = NCA.interpextrapconc(conc[idx], t[idx], x; interpmethod=:linuplogdown)
yli = NCA.interpextrapconc(conc[idx], t[idx], x; interpmethod=:linear)
@test lambdaz(yli, collect(x), idxs=10:20)[1] ≈ lambdaz(ylg, collect(x), idxs=10:20)[1] atol=1e-2

for m in (:linear, :linuplogdown, :linlog)
  @test_broken @inferred auc(conc[idx], t[idx], method=m)
  @test_broken @inferred aumc(conc[idx], t[idx], method=m)
  @test_nowarn NCA.interpextrapconc(conc[idx], t[idx], 1000rand(500), interpmethod=m)
  @test_nowarn auc(conc[idx], t[idx], method=m, interval=(0,100.), auctype=:AUClast)
  @test_nowarn aumc(conc[idx], t[idx], method=m, interval=(0,100.), auctype=:AUMClast)
  # test interval
  aucinf = auc(conc[idx], t[idx])
  aumcinf = aumc(conc[idx], t[idx])
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

@test lambdaz(conc[idx], t[idx], idxs=12:16)[1] == lambdaz(conc[idx], t[idx])[1]
@test lambdaz(conc[idx], t[idx], idxs=12:16)[1] ≈ data[:Lambda_z][1] atol=1e-6
@test log(2)/lambdaz(conc[idx], t[idx])[1] === thalf(conc[idx], t[idx])

fails = (6, 9)
for i in 1:24
  idx = 16(i-1)+1:16*i
  nca = NCAdata(conc[idx], t[idx], dose=dose[i])
  aucs = auc(nca, method=:linear)
  @test aucs === auc(conc[idx], t[idx], dose=dose[i], method=:linear)
  aumcs = aumc(nca, method=:linear)
  @test aumcs === aumc(conc[idx], t[idx], dose=dose[i], method=:linear)
  @test_nowarn auc(conc[idx], t[idx], method=:linuplogdown)
  @test_nowarn aumc(conc[idx], t[idx], method=:linuplogdown)
  @test aucs[2] == aucs[1]/dose[i]
  @test aumcs[2] == aumcs[1]/dose[i]
  if i in fails
    @test_broken data[:AUCINF_obs][i] ≈ aucs[1] atol = 1e-6
    @test_broken data[:AUMCINF_obs][i] ≈ aumcs[1] atol = 1e-6
    @test_broken data[:Lambda_z][i] ≈ lambdaz(nca)[1] atol = 1e-6
  else
    @test data[:AUCINF_obs][i] ≈ aucs[1] atol = 1e-6
    @test data[:AUMCINF_obs][i] ≈ aumcs[1] atol = 1e-6
    @test data[:Lambda_z][i] ≈ lambdaz(nca)[1] atol = 1e-6
  end
  aucs = auc(nca, dose=dose[i], method=:linear, auctype=:AUClast)
  aumcs = aumc(nca, dose=dose[i], method=:linear, auctype=:AUMClast)
  @test aucs[2] == aucs[1]/dose[i]
  @test aumcs[2] == aumcs[1]/dose[i]
  @test data[:AUClast][i] ≈ aucs[1] atol = 1e-6
  @test data[:AUMClast][i] ≈ aumcs[1] atol = 1e-6
  @test aumc_extrap_percent(nca) === aumc_extrap_percent(conc[idx], t[idx])
  @test auc_extrap_percent(nca) === auc_extrap_percent(conc[idx], t[idx])
end
