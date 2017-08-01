import numpy as np
import matplotlib.pyplot as plt

exit_times = np.load('exit_times.npy')

longest = np.amax(exit_times)
fptdist = np.histogram(exit_times, bins=np.arange(longest+1))[0] / 1e5
survival_prob = 1 - np.cumsum(fptdist)


plt.plot(np.arange(longest),survival_prob)
plt.show()


