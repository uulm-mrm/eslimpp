import numpy as np
import subjective_logic as sl
import matplotlib.pyplot as plt


PLOT_EACH_ITER=True
# update_vecs = [np.array([0.7,0.3])] * 3
update_vecs = [
                  np.array([0.7, 0.3]),
                  # # np.array([0.2, 0.8]),
                  # np.array([0.9, 0.1]),
                  # np.array([0.7, 0.3]),
                  # np.array([0.75, 0.25]),
                  # np.array([0.69, 0.31]),
                  # np.array([0.81, 0.19]),
                  # np.array([0.8, 0.2]),
              ]*10
num_updates = len(update_vecs)
alphas = np.array([2.,1.5])

# update_vecs = [
#     np.array([0.7, 0.2, 0.1]),
#     np.array([0.2, 0.7, 0.1]),
#     np.array([0.9, 0.0, 0.1]),
#     np.array([0.7, 0.3, 0.0]),
# ]
# num_updates = len(update_vecs)
# alphas = np.array([4,1.5, 1.0])

dist = sl.DirichletDistribution(alphas)
dist_verify = dist.copy()
mixture = [(1,sl.DirichletDistribution(alphas))]

updated_dist = dist.copy()
zero_update = np.zeros(update_vecs[0].shape)
for num_update, update_vec in enumerate(update_vecs):
    dist_verify.evidences += update_vec
    updated_dist.moment_matching_update_(update_vec)
    new_mixture = []
    normalizer = zero_update.copy()
    for idx in range(2):
        for (factor, distr) in mixture:
            normalizer[idx] += factor * distr.alphas()[idx]

    for (factor, distr) in mixture:
        for idx, value in enumerate(update_vec):
            inc = zero_update.copy()
            inc[idx] = 1
            new_mixture.append((value * factor * distr.alphas()[idx] / normalizer[idx], sl.DirichletDistribution(distr.alphas() + inc)))
    mixture = new_mixture
    print()
    print()
    print('update number:', num_update)
    print(f"update resulted in {len(new_mixture)} mixture components")
    if PLOT_EACH_ITER:
        x_samples = np.linspace(0,1,50)
        orig = np.array(list(map(dist.evaluate, x_samples)))
        update_mixture = np.zeros(orig.shape)
        for factor, distr in mixture:
            update_mixture += factor * np.array(list(map(distr.evaluate, x_samples)))
        update_mm = list(map(updated_dist.evaluate, x_samples))
        update_verify = list(map(dist_verify.evaluate, x_samples))


        plt.plot(orig, color='b', label='before update')
        plt.plot(update_mixture, color='orange', label='mixture')
        plt.plot(update_mm, color='green', label='moment matching')
        plt.plot(update_verify, color='red', label='cbf')

mixture_mean = zero_update.copy()
mixture_var = zero_update.copy()
for factor, distr in mixture:
    mixture_mean += factor * distr.mean()
    mixture_var += factor * distr.variances() + factor * distr.mean()*distr.mean()
mixture_var -= mixture_mean * mixture_mean

print('mean of mixture:', mixture_mean)
print('mean of update:', updated_dist.mean())
print('mean of cbf:', dist_verify.mean())

print('var of mixture:', mixture_var)
print('var of update:', updated_dist.variances())
print('var of cbf:', dist_verify.variances())




if not PLOT_EACH_ITER:
    x_samples = np.linspace(0,1,50)
    orig = np.array(list(map(dist.evaluate, x_samples)))
    update_mixture = np.zeros(orig.shape)
    for factor, distr in mixture:
        update_mixture += factor * np.array(list(map(distr.evaluate, x_samples)))
    update_mm = list(map(updated_dist.evaluate, x_samples))
    update_verify = list(map(dist_verify.evaluate, x_samples))


    plt.plot(orig, color='b', label='before update')
    plt.plot(update_mixture, color='orange', label='mixture')
    plt.plot(update_mm, color='green', label='moment matching')
    plt.plot(update_verify, color='red', label='cbf')
    plt.legend()
plt.show()
