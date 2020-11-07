# ###############################################################
# #
# Kevin Russell #
# ECE 351-51 #
# Lab 10 #
# November 3, 2020 #
# #
# #
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

#%% Part 1

R = 1000
L = 27e-3
C = 100e-9

Hsnum = [1/(R*C),0]
Hsden = [1, 1/(R*C), 1/(L*C)]

steps = 1e3
omega = np.arange(10**3,10**6+steps,steps)
                  

h_mag = (omega/(R*C))/ np.sqrt(((1/(L*C))-omega**2)**2 + (omega/(R*C))**2)

h_phi = np.pi/2 - np.arctan((omega/(R*C))/((1/(L*C))-omega**2))


for i in range (len(omega)):
    if h_phi[i] > np.pi/2:
        h_phi[i] = h_phi[i] - np.pi
    else:
        h_phi[i] = h_phi[i]       
        
        

plt.figure(figsize = (15,10))
plt.subplot (2,1,1)
plt.semilogx(omega, h_mag)
plt.grid()
plt.ylabel('Magnitude of H(s)')
plt.title('Task 1 H(s) Bode Plot')

plt.subplot (2,1,2)
plt.semilogx(omega, h_phi)
plt.grid()
plt.ylabel('Phase of H(s) (rad)')


plt.xlabel('rad/s')
plt.show()

w, bodeMag, bodePhase = sig.bode((Hsnum,Hsden), omega)

plt.figure(figsize = (15,10))
plt.subplot (2,1,1)
plt.semilogx(w, bodeMag)
plt.ylabel('Magnitude of H(s)')
plt.title('Task 2 H(s) Bode Plot')

plt.subplot (2,1,2)
plt.semilogx(w, bodePhase)
plt.grid()
plt.ylabel('Phase of H(S) (deg)')
plt.xlabel('rad/s')
plt.show()

sys = con.TransferFunction(Hsnum, Hsden)
_ = con.bode(sys, omega, dB = True, Hz = True, deg = True, Plot = True)

#%% Part 2
#%%% Task 1

fs = 2*np.pi*50000
steps = 1/fs
t = np.arange(0, 0.01+steps, steps)

x_t = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.figure(figsize = (15,10))
plt.subplot(2,1,1)
plt.plot(t, x_t)
plt.grid()
plt.ylabel('x(t)')
plt.title('Part 2 Signal')

#%%%Task 2

z, p = sig.bilinear(Hsnum, Hsden,4*np.pi*50000) #converts to z domain from s domain

#%%% Task 3

y_t = sig.lfilter(z,p,x_t)

plt.subplot(2,1,2)
plt.plot(t, y_t)
plt.grid()
plt.ylabel('x(t)')
plt.title('Part 2 Filtered Signal')
plt.xlabel('t')
plt.show()

 