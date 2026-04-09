import subjective_logic as sl
import numpy as np


A = sl.Opinion([0.1,0.7]  ,[0.5,0.5])
B = sl.Opinion([0.7,0.1]  ,[0.5,0.5])
C = sl.Opinion([0.2,0.3]  ,[0.5,0.5])
D = sl.Opinion([0.5,0.2]  ,[0.5,0.5])
E = sl.Opinion([0.45,0.45],[0.5,0.5])
F = sl.Opinion([0.,0.]    ,[0.5,0.5])
opinions = [A,B,C,D,E,F]


for opinion in opinions:

    # manual calculation
    bm = np.array(list(opinion.belief_masses))
    prior_bm = np.array(list(opinion.prior_belief_masses))
    uncert = opinion.uncertainty()
    W = 2
    r = W * bm / uncert
    alpha = r + W * prior_bm
    # print(alpha)

    # dirichlet convertion within eSLIM++
    print(opinion.as_dirichlet().alphas())

