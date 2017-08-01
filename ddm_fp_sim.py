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
Pm = Pp
surv_prob_P = np.zeros(Nt)
surv_prob_P[1] = 1 
surv_prob_M = surv_prob_P

# Crank-Nicholson discretization of diffusion
scaling_term = (s * s) / (2 * dx * dx)
D = scaling_term * (np.eye(Nx,k=1) -2 * np.eye(Nx,k=0) + np.eye(Nx,k=-1))  # scaling term times matrix with diagonal -2 and off diagonals 1

# discretization of advection for plus and minus drift
Ap = (-g / dx) * (np.eye(Nx,k=1) - np.eye(Nx,k=0))
Am = -Ap

# create Crank-Nicolson step matrices
diff_inverse = np.linalg.inv(np.eye(Nx) - (dt/2)*D)
Mp = np.dot(diff_inverse, (np.eye(Nx) + dt*Ap + (dt / 2)*D))
Mm = np.dot(diff_inverse, (np.eye(Nx) + dt*Am + (dt / 2)*D))

np.save('Ap.npy',Ap)
np.save('Mp.npy',Mp)

# run CN steps
for j in range(1,Nt):  # j from 1 to Nt-1
    Pp[:,j] = np.dot(Mp, Pp[:,j-1])
    Pm[:,j] = np.dot(Mm, Pm[:,j-1])  # probability
    surv_prob_P[j] = dx * np.sum(Pp[:,j])
    surv_prob_M[j] = dx * np.sum(Pm[:,j])  # survival
    #if j < 100:
        #print surv_prob_P[j]
np.save('prob_diff.npy', Pp)
#plt.figure(1)
plt.pcolormesh(xvec,tvec,Pp.T, cmap='hot', vmin=0, vmax=0.2) # , vmin=0, vmax=0.2

plt.axis('off')
# shading flat  # flat is actually the default shading option
# caxis([0 0.2])
# set(gca,'xtick',[])
# set(gca,'ytick',[])

#plt.figure(2) 
#plt.pcolor(xvec,tvec,Pm.T)
#shading flat, colormap(hot)
#caxis([0 0.2])
#set(gca,'xtick',[])
#set(gca,'ytick',[])

#plt.figure(3)
#plot(tvec,surv_prob_P,'r','linewidth',8)
#set(gca,'xtick',[])
#set(gca,'ytick',[0:3])
#set(gca,'fontsize',30)
#plot(tvec,surv_prob_M,'b','linewidth',8)
#u = log(surv_prob_P / surv_prob_M)  # LLR of survival probabilities
#plot(tvec,u,'k--','linewidth',8)

plt.show()
