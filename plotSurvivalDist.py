import numpy as np
import matplotlib.pyplot as plt

exit_times = np.load('exit_times.npy')
path_data = np.load('path_data.npy')
paths_plus = path_data[0, 1:]
paths_minus = path_data[1, 1:]
paths_pos = path_data[2, 1:]
paths_minus = path_data[3, 1:]

longest = np.amax(exit_times)
fptdist = np.histogram(exit_times, bins=np.arange(longest+1))[0] / 1e5
survival_prob = 1 - np.cumsum(fptdist)

# print path_data
# print np.sum(path_data, 0)

# plt.plot(np.arange(longest),survival_prob)

plt.figure(1)
path_data_labels = ['H+','H-','Pos. LLR','Neg. LLR']
for i in range(4):
    plt.plot(path_data[i,1:], label=path_data_labels[i])
plt.legend()
plt.title('Paths')

plt.figure(2)
plt.plot(np.arange(longest),survival_prob)
plt.title('Survival Prob')


plt.show()


