import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
length = 100
mean1 = 0.1
mean2 = -0.1
var1 = 1
var2 = 1

# # Observations are drawn from the Norm(mean1, var1) distribution.
# obs = np.sqrt(var1) * np.random.randn(length) + mean1  # scale and translate draws from the standard distribution
runs = int(1e5)
max_time = 500
exit_times = np.zeros(runs)


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


def compute_survival_prob(bdy_plus, bdy_minus, num_runs):
    paths_decided = np.zeros(max_time)
    correct = 0

    for r in range(num_runs):
        print r
        ev = 0
        T = 0
        while (ev < bdy_plus) and (ev > bdy_minus) and (T < max_time):
            obs = np.sqrt(var1) * np.random.randn(1) + mean1
            ev += compute_llr(obs, pos.prob, neg.prob)
            T += 1

        if ev >= bdy_plus:
            correct += 1

        paths_decided[T:] += 1
    print "Correct: " + str(100 * correct / runs) + "%"
    return 1 - (paths_decided / num_runs)


a = 3
b = .05

plt.figure(1)
survival_prob_plus = compute_survival_prob(a, -b, runs)
plt.plot(survival_prob_plus, label="H+")

survival_prob_minus = compute_survival_prob(b, -a, runs)
plt.plot(survival_prob_minus, label="H-")
plt.title('Survival Prob Comparison')
plt.legend
plt.xlabel('Time')


plt.figure(2)
nondec_llr = np.log(survival_prob_plus / survival_prob_minus)
plt.plot(nondec_llr)
plt.title("Information from nondecision with " + str(a) + ", -" + str(b))

# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([0, length, 0, 1])

plt.show()

