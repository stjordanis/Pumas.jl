using PuMaS.NCA

df = DataFrame()
# Ref: https://www.boomer.org/c/p4/c05/c05.pdf
df.start_time = [0, 2, 4, 4.5, 6, 8, 10, 12, 18]
df.end_time   = [2, 4, 6, 6.7, 8, 10, 12, 18, 24]
df.volume = [50 , 46 , 48, missing , 49, 46 , 48 , 134, 144]
df.conc = [1.666, 1.069, 0.592, 1000, 0.335, 0.21 , 0.116, 0.047, 0.009]
df.id = 1
df.amt = 0
df.amt[1] = 1
df.route = "ev"
pop = read_nca(df)
@test collect(skipmissing(df.volume)) == pop[1].volume
@test [df.start_time[1:3]; df.start_time[5:end]] == pop[1].start_time
@test [df.end_time[1:3]; df.end_time[5:end]] == pop[1].end_time
@test length(pop[1].conc) == length(pop[1].time) == length(pop[1].start_time) == length(pop[1].end_time) == length(pop[1].volume)
@test_nowarn NCA.auc(pop)
@test_nowarn NCA.lambdaz(pop)
@test NCA.urine_volume(pop)[end][1] == sum(pop[1].volume)
