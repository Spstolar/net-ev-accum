import numpy as np
import matplotlib.pyplot as plt
"""
Modified version of EvAccum.py which simply plots a bunch of trajectories which are stopped when they hit a boundary.
"""

# Parameters for the simulation
length = 100
mean1 = 0.1
mean2 = -0.1
var1 = 1
var2 = 1
bdy_plus = 1
bdy_minus = -3
# # Observations are drawn from the Norm(mean1, var1) distribution.
# obs = np.sqrt(var1) * np.random.randn(length) + mean1  # scale and translate draws from the standard distribution
runs = int(1)
max_time = 200
exit_times = np.zeros(runs)

paths_plus = np.zeros(max_time)  # How many sims have chosen H^+
paths_minus = np.zeros(max_time)  # ^^ H^-
paths_pos = np.zeros(max_time)  # How many sims have not exited and are positive
paths_neg = np.zeros(max_time)  # How many sims have not exited and are negative
correct = 0

class Dist:
    """We define a class for distributions so that we can easily access the truth distributions rather than writing out
    the formula for the distribution each time we want to use it."""
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def prob(self, x):
        return np.exp(-np.power(x - self.mean, 2) / (2*self.var))/(np.sqrt(2 * np.pi * self.var))


pos = Dist(mean1, var1)  # the positive state distribution
neg = Dist(mean2, var2)


def compute_llr(x_array, dist1, dist2):
    """
    Computes the log-likelihood ratio for a given array of observations.
    :param x_array: an array of observations
    :param dist1: the positive truth distribution
    :param dist2: the negative truth distribution
    :return: an array the size of x_array of LLRs
    """
    return np.log(dist1(x_array)/dist2(x_array))

# Compute and store the LLRs as a vector of accumulated evidence.

sims = int(1e4)


for r in range(sims):
    print r
    tracker = np.zeros(max_time+1)
    ev = 0
    T = 0
    time = 0

    while (ev < bdy_plus) and (ev > bdy_minus) and (time < max_time):
        time += 1
        obs = np.sqrt(var1) * np.random.randn(1) + mean1
        ev += compute_llr(obs, pos.prob, neg.prob)
        T += 1
        tracker[time] = ev

    if ev >= bdy_plus:
        tracker[T:] = bdy_plus
    else:
        tracker[T:] = bdy_minus

    plt.plot(tracker, color="red", alpha=0.01)

plt.show()

# The last part here plots time (in steps) against the accumulated evidence. After adding modifications to the plot we
# then call it using the show() method.

# print "Correct: " + str(100 * correct / runs) + "%"
# plt.hist(exit_times, 50, normed=1, facecolor='green', alpha=0.75)
#
# np.save('exit_times.npy', exit_times)
#
# path_data = np.vstack((paths_plus, paths_minus, paths_pos, paths_neg))
# np.save('path_data.npy', path_data)
#
# plt.xlabel('Time')
# plt.ylabel('LLR')
# plt.title('Evidence Accum')
# # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# # plt.axis([0, length, 0, 1])
# # plt.grid(True)
# plt.show()

