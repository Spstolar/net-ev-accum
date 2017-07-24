import numpy as np
import matplotlib.pyplot as plt


length = 100

obs = np.random.randn(length) + 1


class Dist:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def prob(self, x):
        return np.exp(-np.power(x - self.mean, 2) / (2*self.var))/(np.sqrt(2 * np.pi * self.var))

mean1 = 1
mean2 = -1
var1 = 2
var2 = 2

pos = Dist(mean1, var1)  # the positive state distribution
neg = Dist(mean2, var2)


def compute_llr(x_array, dist1, dist2):
    return np.log(dist1(x_array)/dist2(x_array))

llr = compute_llr(obs, pos.prob, neg.prob)
ev = np.cumsum(llr)

plt.plot(np.arange(length), ev)


plt.xlabel('Time')
plt.ylabel('LLR')
plt.title('Evidence Accum')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([0, length, 0, 1])
# plt.grid(True)
plt.show()

