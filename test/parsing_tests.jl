using PKPDSimulator, Base.Test

covariates = [1]
dvs = [1]
z = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/event_data/ev1.csv"), covariates,dvs,
              separator=',')

@test z[1].event_times == collect(0:12.0:36.0)
for ev in z[1].events
  @test ev == z[1].events[1]
  @test ev.cmt == 1
end
