using PuMaS.NCA

df = DataFrame()
# Ref: https://www.boomer.org/c/p4/c05/c05.pdf
df.start_time = [0, 2, 4, 6, 8, 10, 12, 18]
df.end_time   = [2, 4, 6, 8, 10, 12, 18, 24]
df.volume = [50 , 46 , 48 , 49 , 46 , 48 , 134, 144]
df.conc = [1.666, 1.069, 0.592, 0.335, 0.21 , 0.116, 0.047, 0.009]
df.id = 1
df.amt = 0
df.amt[1] = 1
df.route = "ev"
pop = read_nca(df)
@test_nowarn NCA.auc(pop)
