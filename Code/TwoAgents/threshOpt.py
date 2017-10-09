import numpy as np
import matplotlib.pyplot as plt
"""
We use the direct simulation as in evAccum.py
Here we have two agents with unidirectional coupling to compute the optimal amount of information to integrate when the
sending agent makes a decision.
"""

# Parameters for the simulation
length = 100
mean1 = 0.1
mean2 = -0.1
var1 = 1
var2 = 1
p_correct = .7
bdy_plus = np.log(p_correct / (1 - p_correct))
print bdy_plus
bdy_minus = - bdy_plus

# # Observations are drawn from the Norm(mean1, var1) distribution.
# obs = np.sqrt(var1) * np.random.randn(length) + mean1  # scale and translate draws from the standard distribution
runs = int(1e5)
max_time = 500
exit_times = np.zeros(runs)

paths_plus = np.zeros(max_time)  # How many sims have chosen H^+
paths_minus = np.zeros(max_time)  # ^^ H^-
paths_pos = np.zeros(max_time)  # How many sims have not exited and are positive
paths_neg = np.zeros(max_time)  # How many sims have not exited and are negative

side_thresholds = 10
total_thresholds = 2 * side_thresholds
buffer = .2
inc = buffer / side_thresholds
thresholds = np.hstack((np.arange(bdy_plus - buffer, bdy_plus, inc), np.arange(bdy_plus, bdy_plus + buffer, inc)))
correct = np.zeros(total_thresholds)

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
for index, thresh in enumerate(thresholds):
    for r in range(runs):
        # rest evidence for the new trial
        ev_1 = 0
        ev_2 = 0
        dec_1 = 0
        time = 0

        while (ev_2 < bdy_plus) and (ev_2 > bdy_minus) and (time < max_time):  # run until boundary crossed or time out
            # undecided agents make observations and compute LLRs
            if dec_1 == 0:
                obs_1 = np.sqrt(var1) * np.random.randn(1) + mean1
                ev_1 += compute_llr(obs_1, pos.prob, neg.prob)

            obs_2 = np.sqrt(var1) * np.random.randn(1) + mean1
            ev_2 += compute_llr(obs_2, pos.prob, neg.prob)

            if ev_1 >= bdy_plus:
                ev_2 += thresh
                dec_1 = 1
                ev_1 = 0

            if ev_1 <= bdy_minus:
                ev_2 -= thresh
                dec_1 = -1
                ev_1 = 0

            time += 1

        if ev_2 >= bdy_plus:
            correct[index] += 1




# The last part here plots time (in steps) against the accumulated evidence. After adding modifications to the plot we
# then call it using the show() method.

correct_probs = correct / float(runs)
print "Correct: " + str(correct_probs)

plt.plot(thresholds, correct_probs)
# plt.hist(exit_times, 50, normed=1, facecolor='green', alpha=0.75)
#
# np.save('exit_times.npy', exit_times)
#
# path_data = np.vstack((paths_plus, paths_minus, paths_pos, paths_neg))
# np.save('path_data.npy', path_data)
#
plt.xlabel('Jump Amount')
plt.ylabel('Probability Second Agent Correct')
plt.title('Optimality of Threshold Jump')
# # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# # plt.axis([0, length, 0, 1])
# # plt.grid(True)
plt.show()

