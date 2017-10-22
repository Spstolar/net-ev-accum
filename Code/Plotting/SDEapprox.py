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
time = np.arange(0, 1, 1.0/length)
sub = np.zeros(10)
subtime = np.zeros(10)
for i in np.arange(9, 100, 10):
    print i
    print i/10
    sub[i/10] = ev[i]
    subtime[i/10] = time[i]

# The last part here plots time (in steps) against the accumulated evidence. After adding modifications to the plot we
# then call it using the show() method.
plt.scatter(subtime, sub)
plt.scatter(time, llr, color='orange')


plt.xlabel('Time')
plt.ylabel('LLR')
plt.title('Evidence Accumulation')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([0, 1, bdy_minus, bdy_plus])
# plt.grid(True)
plt.show()

