# ###############################################################
# #
# Kevin Russell #
# ECE 351-51 #
# Lab 5 #
# September 29, 2020 #
# #
# #
# ###############################################################

#%% Part 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

R = 1000
L = 27e-3
C = 100e-9

def step(t): #defining step function using mathematical definition
    y=np.zeros(t.shape)
    
    for i in range (len(t)):
        if t[i]<0:
            y[i]=0
        else:
            y[i]=1
            
    return y 

def sine(R,L,C, t):
    
    alpha = -1 / (2*R*C)
    omega = 0.5*np.sqrt((1/(R*C))**2-4*(1/(np.sqrt(L*C)))**2 + 0*1j)
    p = alpha + omega
    
    g=(1/(R*C))*p
    
    gmag = np.abs(g)
    gang = np.angle(g)
    
    y=(gmag/np.abs(omega))*np.exp(alpha * t)*np.sin(np.abs(omega) * t + gang)
        
    return y*step(t)

plt.rcParams.update({'font.size':14})


steps =1e-5
t=np.arange(0,1.2e-3+steps,steps)

hand=sine(R,L,C,t)

plt.figure(figsize = (10,20))
plt.subplot(2,1,1)
plt.plot(t,hand)
plt.grid()
plt.ylabel('h(t), hand')
plt.title('Impulse response H(t)')

num = [(1/(R*C)),0]
den = [1, (1/(R*C)), (1/(L*C))]

tout, yout = sig.impulse((num,den), T = t)

plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('h(t), python')



plt.xlabel('Time')
plt.show()


#%%Part 2

tout, yout = sig.step((num,den), T=t)

plt.figure(figsize = (10,20))
plt.plot(tout, yout)
plt.grid()
plt.ylabel('h(t)')
plt.title('Step response of H(s)')


plt.xlabel('Time')
plt.show()


## This result is like it is a shifted sinusoidal from the Part 1 Task 2 plot, 
## starting at the 0 point on the second graph.  This makes sense becasue the 
## step response is a convolution between H(s) and u(s).The graph looks shifted 
## and starts at 0 becasue it has to account for the initial "build-up" of area 
## when the convolution first begins, thinking of it graphical sense.
