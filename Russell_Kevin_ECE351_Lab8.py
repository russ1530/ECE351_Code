# ###############################################################
# #
# Kevin Russell #
# ECE 351-51 #
# Lab 8 #
# October 20, 2020 #
# #
# #
# ###############################################################

#%% Part 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def a(k):
    
   return 0
    
print ('a(0)=',a(0))
print ('a(1)=',a(1))

def b(k):
    
    b=(2-2*np.cos(np.pi*k))/(np.pi*k)
    
    return b



for i in range (1,4):
    print ('b(',i,')=',b(i))
    

T=8
omega=(2*np.pi)/T

def FS(t,T,k):
    x=0
    omega = (2*np.pi)/T
    for j in range (1,k+1):
        x = x + b(j)*np.sin(j*omega*t)
        
    return x

steps =1e-5
t=np.arange(0,20+steps,steps)

X1 = FS(t,8,1)
X2 = FS(t,8,3)
X3 = FS(t,8,15)
X4 = FS(t,8,50)
X5 = FS(t,8,150)
X6 = FS(t,8,1500)

plt.figure(figsize = (10,12))
plt.subplot(3,1,1)
plt.plot(t, X1)
plt.grid()
plt.ylabel('N=1')
plt.title('Fourier Series Approximation of a Square Wave')

plt.subplot(3,1,2)
plt.plot(t, X2)
plt.grid()
plt.ylabel('N=3')

plt.subplot(3,1,3)
plt.plot(t, X3)
plt.grid()
plt.ylabel('N=15')

plt.xlabel('Time(s)')
plt.show()


plt.figure(figsize = (10,12))
plt.subplot(3,1,1)
plt.plot(t, X4)
plt.grid()
plt.ylabel('N=50')
plt.title('Fourier Series Approximation of a Square Wave')

plt.subplot(3,1,2)
plt.plot(t, X5)
plt.grid()
plt.ylabel('N=150')

plt.subplot(3,1,3)
plt.plot(t, X6)
plt.grid()
plt.ylabel('N=1500')

plt.xlabel('Time(s)')
plt.show()


    
        
        