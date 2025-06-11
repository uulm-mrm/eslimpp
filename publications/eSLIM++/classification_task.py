import numpy as np
import subjective_logic as sl
import matplotlib.pyplot as plt

from termcolor import colored

# measurement used for update
update_vec = np.array([0.6,0.4])
# initial alphas of Dirichlet distribution prior to the update
alphas = np.array([1.5,1.5])

dist = sl.DirichletDistribution(alphas)
dist_verify = dist.copy()
mixture = [(1,sl.DirichletDistribution(alphas))]

updated_dist = dist.copy()
zero_update = np.zeros(update_vec.shape)

############################
# begin update

# cbf
dist_verify.evidences += update_vec
# moment matching
updated_dist.moment_matching_update_(update_vec)
#mixture
new_mixture = []
# valid for a single update
for (factor, distr) in mixture:
    for idx, value in enumerate(update_vec):
        inc = zero_update.copy()
        inc[idx] = 1
        new_mixture.append((factor * value, sl.DirichletDistribution(distr.alphas() + inc)))
mixture = new_mixture

############################
# end update

# calc mean and variance of mixture
mixture_mean = zero_update.copy()
mixture_var = zero_update.copy()
for factor, distr in mixture:
    mixture_mean += factor * distr.mean()
    mixture_var += factor * distr.variances() + factor * distr.mean()*distr.mean()
mixture_var -= mixture_mean * mixture_mean

print(f'mean: {mixture_mean} | {colored('mixture', 'yellow')}')
print(f'mean: {updated_dist.mean()} | {colored('moment matching update', 'green')}')
print(f'mean: {dist_verify.mean()} | {colored('cbf','red')}')
print()
print(f'var: {mixture_var} | {colored('mixture', 'yellow')}')
print(f'var: {updated_dist.variances()} | {colored('moment matching update','green')}')
print(f'var: {dist_verify.variances()} | {colored('cbf','red')}')
print()
print(f'alphas {dist.alphas()} | {colored('before update','blue')}')
print(f'{colored('mixture', 'yellow')} components:')
for factor, distr in mixture:
    print(f'\tfactor: {factor}, alphas: {distr.alphas()}')
print(f'alphas: {updated_dist.alphas()} | {colored('moment matching update','green')}')
print(f'alphas: {dist_verify.alphas()} | {colored('cbf','red')}')


# sample distrs for plotting
x_samples = np.linspace(0,1,100)
orig = np.array(list(map(dist.evaluate, x_samples)))
update_mixture = np.zeros(orig.shape)
for factor, distr in mixture:
    update_mixture += factor * np.array(list(map(distr.evaluate, x_samples)))
update_mm = list(map(updated_dist.evaluate, x_samples))
update_verify = list(map(dist_verify.evaluate, x_samples))

# plot distributions
plt.plot(orig, color='b', label='before update')
plt.plot(update_mixture, color='orange', label='mixture')
plt.plot(update_mm, color='green', label='moment matching')
plt.plot(update_verify, color='red', label='cbf')
plt.legend()
plt.show()
