using Pumas, Test

df = DataFrame()
# Ref: https://www.boomer.org/c/p4/c05/c05.pdf
amtu=u"mg"; concu=u"mg/mL"; volumeu=u"mL"; timeu=u"hr"
df[!,:start_time] = [0, 2, 4, 4.5, 6, 8, 10, 12, 18]
df[!,:end_time]   .= [2, 4, 6, 6.7, 8, 10, 12, 18, 24]
df[!,:volume] .= [50 , 46 , 48, missing , 49, 46 , 48 , 134, 144]
df[!,:conc] .= [1.666, 1.069, 0.592, 1000, 0.335, 0.21 , 0.116, 0.047, 0.009]
df[!,:id] .= 1
df[!,:amt] .= 0.0
df.amt[1] = 300
df[!,:route] .= "iv"
pop = read_nca(df, amtu=amtu, concu=concu, volumeu=volumeu, timeu=timeu)
@test collect(skipmissing(df.volume))*volumeu == pop[1].volume
@test [df.start_time[1:3]; df.start_time[5:end]]*timeu == pop[1].start_time
@test [df.end_time[1:3]; df.end_time[5:end]]*timeu == pop[1].end_time
@test length(pop[1].conc) == length(pop[1].time) == length(pop[1].start_time) == length(pop[1].end_time) == length(pop[1].volume)
@test ustrip(NCA.aurc(pop, auctype=:last)[!,2][1]) ≈ 180.38833333333333
@test_nowarn NCA.lambdaz(pop)
@test_nowarn NCA.tmax_rate(pop)[!, 2][1], NCA.max_rate(pop)[!, 2][1], NCA.mid_time_last(pop)[!, 2][1], NCA.rate_last(pop)[!, 2][1], NCA.aurc(pop)[!, 2][1], NCA.aurc_extrap_percent(pop)[!, 2][1]
@test NCA.urine_volume(pop)[!, end][1] == sum(pop[1].volume)
@test NCA.rate_last(pop)[!, 2][1] != NCA.rate_last(pop, pred=true)[!, 2][1]

report = NCAReport(pop)
reportdf = NCA.to_dataframe(report)
