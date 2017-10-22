import numpy as np
import matplotlib.pyplot as plt


## Params
length = 10
lam = 15
mean = 1
variance = 2
draws = int(1e4)

gauss_sums = np.zeros(draws)

for i in range(draws):
    obs = np.sqrt(variance) * np.random.randn(length) + mean
    total = np.sum(obs) / length + np.sum(( obs - mean ) / np.sqrt(length))
    gauss_sums[i] = total


plt.hist(gauss_sums)


# plt.xlabel('Observation')
# plt.ylabel('Probability of Observation')
# plt.title('Observation Distribution for $H^+$')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

# For dimensions
# y_min = np.amin(y_vals)
# y_max = np.amax(y_vals) + .1
#
# plt.axis([x_vals[0], x_vals[-1], y_min, y_max])
# plt.grid(True)
plt.show()