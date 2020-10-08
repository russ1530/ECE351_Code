# ###############################################################
# #
# Kevin Russell #
# ECE 351-51 #
# Lab 6 #
# October 6, 2020 #
# #
# #
# ###############################################################

#%% Part 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def step(t): #defining step function using mathematical definition
    y=np.zeros(t.shape)
    
    for i in range (len(t)):
        if t[i]<0:
            y[i]=0
        else:
            y[i]=1
            
    return y 

def response(t):
    
    return (0.5 + np.exp(-6*t) - 0.5*np.exp(-4*t))*step(t)



steps =1e-5
t=np.arange(0,2+steps,steps)

y = response(t)

plt.figure(figsize = (10,20))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t), hand')
plt.ylim([0,1])
plt.title('Step response y(t)')

num = [1,6,12]
den = [1, 10,24]

tout, yout = sig.step((num,den), T = t)


plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('y(t), python')
plt.ylim([0,1])



plt.xlabel('Time')
plt.show()


num = [1,6,12]
den = [1, 10,24,0]

R, P, _ = sig.residue(num,den)

print (R,P)



#%%Part 2


num = [25250]
den = [1,18,218,2036,9085,25250,0]

R, P, _ = sig.residue(num,den)

print (R,P)

y=0

def cosine(R,P,t):
    omega = np.imag(P)
    alpha = np.real(P)
    kmag = np.abs(R)
    kangle = np.angle(R)
    
    y = kmag*np.exp(alpha*t)*np.cos(omega*t+kangle)
    
    
    return y*step(t)

steps =1e-5
t=np.arange(0,4.5+steps,steps)

length =len(R)

for i in range(length):
    y+=cosine(R[i],P[i],t)



plt.figure(figsize = (10,20))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t), residue')
plt.title('Step Response y(t)')


num = [25250]
den = [1,18,218,2036,9085,25250]
tout, yout = sig.step((num,den), T = t)


plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('y(t), Scipy Step')



plt.xlabel('Time')
plt.show()