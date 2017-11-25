using PKPDSimulator, Base.Test

covariates = [1]
dvs = [1]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/ev1.csv"), covariates,dvs,
              separator=',')

@test map(x->x.time,data[1].event_times) == collect(0:12.0:36.0)
for ev in data[1].events
  @test ev == data[1].events[1]
  @test ev.cmt == 1
end
