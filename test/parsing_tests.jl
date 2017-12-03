using PKPDSimulator, Base.Test

covariates = []
dvs = []
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/ev1.csv"), covariates,dvs,
              separator=',')

@test map(x->x.time,data[1].event_times) == collect(0:12.0:36.0)
for ev in data[1].events
  @test ev == data[1].events[1]
  @test ev.cmt == 1
end

using DataFrames
gen_data = build_dataset(amt=100, addl=3, cmt=1, ii=12, ss=0)
@test data[1].id == gen_data.id
@test data[1].obs == gen_data.obs
@test data[1].obs_times == gen_data.obs_times
@test data[1].z == gen_data.z
@test data[1].event_times == gen_data.event_times
@test data[1].events == gen_data.events
