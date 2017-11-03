import numpy as np
import matplotlib.pyplot as plt

"""
Adapted from simddmfp.m
Simulates the Fokker-Planck equation for drift diffusion.
Outputs:
    1. The probability distribution under the H^+ truth assumption.
    2. ^^ for H^-
    3. The survival probabilities and their log ratio.
    4. The positive and negative path probabilities and the information gained when you know an agent has positive LLR.
"""

# model params
g = 1  # drift
s = 1  # variance
hp = 3  # upper threshold
hm = -1.0  # lower threshold
L = hp-hm  # length of the domain

# discretization parameters
dx = 0.01
Nx = int(L / dx + 1)
xvec = np.linspace(hm, hp, Nx)  # split the x-axis into dx sized bins
T = 10  # total time to run
dt = 0.001 
Nt = int(T / dt + 1)
tvec = np.linspace(0, T, Nt)  # split time into dt sized bins

# setup probability and survival arrays
Pp = np.zeros((Nx, Nt))
zero_ind = np.argmin(abs(xvec))  # grab the index of the zero or the xvalue closest to zero
Pp[zero_ind, 0] = 1.0 / dx  # we want probability to be concentrated at the zero cell, so it should have area 1
Pm = np.copy(Pp)
surv_prob_P = np.zeros(Nt)
surv_prob_P[0] = 1 
surv_prob_M = np.copy(surv_prob_P)
pos_surv_pos = np.zeros(Nt)  # probability mass above zero but not escaping given H^+
pos_surv_neg = np.zeros(Nt)  # ^^ for H^-

# Crank-Nicolson discretization of diffusion
scaling_term = (s * s) / (2 * dx * dx)
D = scaling_term * (np.eye(Nx, k=1) - 2 * np.eye(Nx, k=0) + np.eye(Nx, k=-1))  # scalar * (diagonal -2, off diagonals 1)

# discretization of advection for plus and minus drift
Ap = (-g / dx) * (np.eye(Nx, k=1) - np.eye(Nx, k=0))
Am = -np.copy(Ap)

# create Crank-Nicolson step matrices
diff_inverse = np.linalg.inv(np.eye(Nx) - (dt/2)*D)
Mp = np.dot(diff_inverse, (np.eye(Nx) + dt*Ap + (dt / 2)*D))
Mm = np.dot(diff_inverse, (np.eye(Nx) + dt*Am + (dt / 2)*D))

# run CN steps
for j in range(1, Nt):  # j from 1 to Nt-1
    Pp[:, j] = np.dot(Mp, Pp[:, j-1])
    Pm[:, j] = np.dot(Mm, Pm[:, j-1])  # probability
    surv_prob_P[j] = dx * np.sum(Pp[:, j])
    surv_prob_M[j] = dx * np.sum(Pm[:, j])  # survival
    right_pos = dx * np.sum(Pp[zero_ind:, j])
    right_neg = dx * np.sum(Pm[zero_ind:, j])
    pos_surv_pos[j] = right_pos / surv_prob_P[j]
    pos_surv_neg[j] = right_neg / surv_prob_M[j]


np.save('surv_P.npy', Pp)
np.save('surv_M.npy', Pm)

plt.figure(1)
plt.pcolormesh(xvec, tvec, Pp.T, cmap='hot', vmin=0, vmax=0.2)  # pcolormesh is much faster than pcolor for large arrays
plt.axvline(x=0)
plt.axis('off')

f = plt.figure()
# f.savefig("fp_plus.pdf", bbox_inches='tight')
# shading flat  # flat is actually the default shading option

plt.figure(2)
plt.pcolormesh(xvec, tvec, Pm.T, cmap='hot', vmin=0, vmax=0.2)
plt.axvline(x=0)
plt.axis('off')

plt.figure(3)
plt.plot(tvec, surv_prob_P, color='red', linewidth=8, label=r'S_+(t)')  # survival prob H^+
plt.plot(tvec, surv_prob_M, color='blue', linewidth=8, label=r'S_-(t)')  # survival prob H^-
u = np.log(surv_prob_P / surv_prob_M)
max_info = np.amax(u)
y_max = np.amax((max_info, 1)) + 0.5
plt.plot(tvec, u, 'k--', linewidth=8, label='Non-decision Evidence')  # non-decision information
plt.xticks([0, T/2, T])
plt.yticks([0, 1, max_info])
plt.xlabel('Time')
plt.ylabel('Probability/Evidence')
plt.title('Survival Probabilities and Non-decision Evidence')
plt.legend()

plt.figure(4)
plt.plot(tvec, pos_surv_pos, color='orange', linewidth=2, label='R+')
plt.plot(tvec, pos_surv_neg, color='blue', linewidth=2, label='R-')
r_prob_ratio = np.log(pos_surv_pos / pos_surv_neg)  # log R_+ / R_-  clique info
plt.plot(tvec, r_prob_ratio, color='purple', linewidth=4, label='Info')
plt.legend()

plt.show()
