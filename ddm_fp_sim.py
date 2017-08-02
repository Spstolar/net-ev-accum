# code adapted from simddmfp.m
# simulate fokker planck equation for drift diffusion.
import numpy as np
import matplotlib.pyplot as plt

# model params
g = 1  # drift
s = 1  # variance
hp = 4  # upper threshold
hm = -1  # lower threshold
L = hp-hm  # length of the domain

# discretization parameters
dx = 0.01
Nx = int(L / dx + 1)
xvec = np.linspace(hm,hp,Nx)  # split the x-axis into dx sized bins
T = 10  # total time to run
dt = 0.001 
Nt = int(T / dt + 1)
tvec = np.linspace(0,T,Nt)  # split time into dt sized bins

# setup probability and survival arrays
Pp = np.zeros((Nx,Nt)) 
zero_ind = np.argmin(abs(xvec))  # grab the index of the zero or the xvalue closest to zero
Pp[zero_ind,0] = 1.0 / dx  # we want probability to be concentrated at the zero cell, so it should have area 1
Pm = np.copy(Pp)
surv_prob_P = np.zeros(Nt)
surv_prob_P[0] = 1 
surv_prob_M = np.copy(surv_prob_P)

# Crank-Nicolson discretization of diffusion
scaling_term = (s * s) / (2 * dx * dx)
D = scaling_term * (np.eye(Nx,k=1) -2 * np.eye(Nx,k=0) + np.eye(Nx,k=-1))  # scaling term times matrix with diagonal -2 and off diagonals 1

# discretization of advection for plus and minus drift
Ap = (-g / dx) * (np.eye(Nx,k=1) - np.eye(Nx,k=0))
#Am = (-g / dx) * (np.eye(Nx,k=-1) - np.eye(Nx,k=0))
Am = -np.copy(Ap)

# create Crank-Nicolson step matrices
diff_inverse = np.linalg.inv(np.eye(Nx) - (dt/2)*D)
Mp = np.dot(diff_inverse, (np.eye(Nx) + dt*Ap + (dt / 2)*D))
Mm = np.dot(diff_inverse, (np.eye(Nx) + dt*Am + (dt / 2)*D))

# run CN steps
for j in range(1,Nt):  # j from 1 to Nt-1
    Pp[:,j] = np.dot(Mp, Pp[:,j-1])
    Pm[:,j] = np.dot(Mm, Pm[:,j-1])  # probability
    surv_prob_P[j] = dx * np.sum(Pp[:,j])
    surv_prob_M[j] = dx * np.sum(Pm[:,j])  # survival

plt.figure(1)
plt.pcolormesh(xvec, tvec, Pp.T, cmap='hot', vmin=0, vmax=0.2) #pcolormesh is much faster than pcolor in python for large arrays
plt.axis('off')
# shading flat  # flat is actually the default shading option


plt.figure(2) 
plt.pcolormesh(xvec, tvec, Pm.T, cmap='hot', vmin=0, vmax=0.2)
plt.axis('off')


plt.figure(3)
plt.plot(tvec, surv_prob_P, color='red', linewidth=8)
plt.plot(tvec, surv_prob_M, color='blue', linewidth=8)
u = np.log( surv_prob_P / surv_prob_M)
plt.plot(tvec, u, 'k--', linewidth=8)
plt.axis([0, T, 0, 3])
#set(gca,'xtick',[])
#set(gca,'ytick',[0:3])
#set(gca,'fontsize',30)


plt.show()
