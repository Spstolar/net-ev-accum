import numpy as np
import matplotlib.pyplot as plt
"""
This handles simulation of the evidence accumulation process directly. An agent makes a predefined number of
observations and the derived information is computed exactly, rather than being approximated with a FP-solution.
"""


# Parameters for the simulation
length = 100
mean1 = 0.1
mean2 = -0.1
var1 = 1
var2 = 1
bdy_plus = 3
bdy_minus = -3

# Observations are drawn from the true Norm(mean1, var1) distribution.
obs = np.sqrt(var1) * np.random.randn(length) + mean1  # scale and translate draws from the standard distribution


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

llr = compute_llr(obs, pos.prob, neg.prob)
ev = np.cumsum(llr)

done = 0
for i in range(ev.size):
    if done == 0:
        if ev[i] >= bdy_plus:
            ev[i] = bdy_plus
            done = 1
    else:
        ev[i] = bdy_plus

# The last part here plots time (in steps) against the accumulated evidence. After adding modifications to the plot we
# then call it using the show() method.
plt.plot(np.arange(length), ev)

plt.xlabel('Time')
plt.ylabel('LLR')
plt.title('Evidence Accumulation')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([0, length, bdy_minus - 0.1, bdy_plus + 0.1])
# plt.grid(True)
plt.show()

