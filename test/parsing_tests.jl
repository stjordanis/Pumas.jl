using PuMaS, Test

@testset "nmtran" begin
  data = process_nmtran(example_nmtran_data("event_data/data1"))
  @test_nowarn show(data)

  @test getproperty.(data[1].events, :time) == 0:12:36

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
  @testset "Dosage Regimen" begin
    gen_subject = Subject(evs = DosageRegimen(100, ii = 12, addl = 3))
    @test data[1].id == gen_subject.id
    @test data[1].covariates == gen_subject.covariates
    @test data[1].events == gen_subject.events
  end
end
@testset "Time Variant Covariates" begin
  data = process_nmtran(example_nmtran_data("time_variant_cvs"), [:weight, :dih])
  @test data[1].covariates.weight |> (x -> isa(x, Vector{Int}) && length(x) == 9)
  @test data[1].covariates.dih == 2
end
