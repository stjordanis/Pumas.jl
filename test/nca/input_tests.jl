using PuMaS.NCA, Test

@test_nowarn NCA.checkconctime([1,2,3,4], 1:4)
@test_nowarn NCA.checkconctime([1,2,missing,4], 1:4)
@test_throws ArgumentError NCA.checkconctime([1,2,missing,4], 1:5)

@test_logs (:warn, "No concentration data given") NCA.checkconctime(Int[])
@test_logs (:warn, "No time data given") begin
  @test_throws ArgumentError NCA.checkconctime([1,2], Int[])
end
@test_throws ArgumentError NCA.checkconctime(Set([1,2]))
@test_throws ArgumentError NCA.checkconctime([1,2], Set([1,2]))
@test_throws ArgumentError NCA.checkconctime([missing, true])
@test_logs (:warn, "All concentration data is missing") NCA.checkconctime([missing, missing])
@test_logs (:warn, "Negative concentrations found") NCA.checkconctime([missing, -1])

@test_throws ArgumentError NCA.checkconctime([1,2], [missing, 2])
@test_throws ArgumentError NCA.checkconctime([1,2], Set([1, 2]))
@test_throws ArgumentError NCA.checkconctime([1,2], [2, 1])
@test_throws ArgumentError NCA.checkconctime([1,2], [1, 1])

conc, t = NCA.cleanmissingconc([1,2,missing,4], 1:4)
@test conc == [1,2,4]
@test eltype(conc) === Int
@test t == [1,2,4]

conc, t = NCA.cleanmissingconc([1,2,missing,4], 1:4, missingconc=100)
@test conc == [1,2,100,4]
@test eltype(conc) === Int
@test t === 1:4

conc, t = NCA.cleanblq(zeros(6), 1:6, concblq=:drop)
@test isempty(conc)
@test isempty(t)
conc, t = NCA.cleanblq(zeros(6), 1:6, concblq=:keep)
@test conc == zeros(6)
@test t == 1:6
conc, t = NCA.cleanblq(ones(6), 1:6, concblq=:drop)
@test conc == ones(6)
@test t == 1:6
conc, t = NCA.cleanblq([0,1,1,3,0], 1:5, concblq=Dict(:first=>:drop, :middle=>:keep, :last=>:drop))
@test conc == [1,1,3]
@test t == 2:4

@test_nowarn show(NCAdata([1,2,3.], 1:3))
