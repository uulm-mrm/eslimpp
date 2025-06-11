#!/usr/bin/python3
import numpy as np
import subjective_logic as sl
import matplotlib.pyplot as plt


update_vec = np.array([0.8,0.2])

alphas = np.array([1,3])
dist = sl.DirichletDistribution2d(alphas)
dist_x0 = sl.DirichletDistribution2d(alphas + np.array([1,0]))
dist_x1 = sl.DirichletDistribution2d(alphas + np.array([0,1]))
dist_verify = sl.DirichletDistribution2d(alphas + update_vec)

updated_dist = dist.moment_matching_update(update_vec)
print(dist.alphas())
print(updated_dist.alphas())

print('mean of mixture: ', (update_vec[0] * dist_x0.mean() + update_vec[1] * dist_x1.mean()))
print('mean of update:', updated_dist.mean())


orig = list(map(dist.evaluate, np.linspace(0,1,50)))
update_x0 = update_vec[0] * np.array(list(map(dist_x0.evaluate, np.linspace(0,1,50))))
update_x1 = update_vec[1] * np.array(list(map(dist_x1.evaluate, np.linspace(0,1,50))))
update_mm = list(map(updated_dist.evaluate, np.linspace(0,1,50)))
update_verify = list(map(dist_verify.evaluate, np.linspace(0,1,50)))


plt.plot(orig, color='b', label='before update')
# plt.plot(update_x0)
# plt.plot(update_x1)
plt.plot(update_x0 + update_x1, color='orange', label='mixture')
plt.plot(update_mm, color='green', label='moment matching')
plt.plot(update_verify, color='red', label='cbf')
plt.legend()
plt.show()