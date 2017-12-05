using PKPDSimulator, Base.Test

covariates = []
dvs = []
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/ev1.csv"), covariates,dvs,
              separator=',')

@test map(x->x.time,data[1].events) == collect(0:12.0:36.0)
for ev in data[1].events
  @test ev.amt == data[1].events[1].amt
  @test ev.evid == data[1].events[1].evid
  @test ev.cmt == data[1].events[1].cmt
  @test ev.rate == data[1].events[1].rate
  @test ev.ss == data[1].events[1].ss
  @test ev.ii == data[1].events[1].ii
  @test ev.rate_dir == data[1].events[1].rate_dir
  @test ev.cmt == 1
end

gen_data = build_dataset(amt=100, addl=3, cmt=1, ii=12, ss=0)
@test data[1].id == gen_data.id
@test data[1].obs == gen_data.obs
@test data[1].obs_times == gen_data.obs_times
@test data[1].z == gen_data.z
@test data[1].events == gen_data.events
