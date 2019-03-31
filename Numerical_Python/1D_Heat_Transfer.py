import numpy as np
import matplotlib.pyplot as plt

L = 0.5 #m 
n = 10
T0 = 293.0 #K
T1s = 77.15 #K
T2s = 77.15 #K
dx = L/n
k = 413 #J/(m s K)
r = 8.96*10**3 #kg/m**3 * 10**-3 kg/g * 10**6 cm**3/m**3 
cp = 390 #J/(kg K)
a = k/(r*cp)
t_f = 1800
dt = 5

x = np.linspace(dx/2, L-dx/2, n)
T = np.ones(n)*T0
dTdt = np.empty(n)
t = np.arange(0,t_f, dt)


for j in range(1,len(t)):
    plt.clf()
    for i in range(1,n-1):
        dTdt[i] = a * (-(T[i] - T[i-1])/dx**2 + (T[i+1] - T[i])/dx**2)
    dTdt[0] = a * (-(T[0] - T1s)/dx**2 + (T[1] - T[0])/dx**2)
    dTdt[n-1] = a * (-(T[n-1] - T[n-2])/dx**2 + (T2s - T[n-1])/dx**2)
    T = T + dTdt*dt
    plt.figure(1)
    plt.plot(x,T,label = "dt={}".format(t[j]))
    plt.axis([0,L,0,300])
    plt.legend()
    plt.show()
    plt.pause(0.1)
