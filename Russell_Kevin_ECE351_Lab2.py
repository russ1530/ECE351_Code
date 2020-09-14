# ###############################################################
# #
# Kevin Russell #
# ECE 351-51 #
# Lab 2 #
# September 8, 2020 #
# #
# #
# ###############################################################

#%%EXAMPLE

"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':14})

steps =1e-2
t=np.arange(0,5+steps,steps)

print('number of elements: len(t) = ', len(t), '\nFirst Element: t[0] = ', t[0], ' \nLast Element: t[len(t) -1] = ', t[len(t) - 1])

def example1(t):
    y=np.zeros(t.shape)
    
    for i in range (len(t)):
        if i<(len(t)+1)/3:
            y[i] =t[i]**2
            
        else:
            y[i]= np.sin(5*t[i])+2
    return y

y=example1(t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t) with good resolution')
plt.title('background -illustration of for loops and if/else statements')

t=np.arange(0,5+0.25, 0.25)
y=example1(t)
plt.subplot(2,1,2)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t) with poor resolution')
plt.xlabel('t')
plt.show()
"""
#%% Part 1

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':14})

steps =1e-2
t=np.arange(0,10+steps,steps)

def func1(t):
    y=np.zeros(t.shape)
    
    for i in range (len(t)):
        y[i]=np.cos(t[i])
    return y

y=func1(t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t) with good resolution')
plt.xlabel('Time')
plt.title('Part 1: Task 2 - Cosine Plot')
plt.show()

#%% Part 2
def step(t): #defining step function using mathematical definition
    y=np.zeros(t.shape)
    
    for i in range (len(t)):
        if t[i]<0:
            y[i]=0
        else:
            y[i]=1
            
    return y    

def ramp(t): #defining ramp function using mathematical definition
    y=np.zeros(t.shape)
    
    for i in range (len(t)):
        if t[i]<0:
            y[i]=0
        else:
            y[i]=t[i]
    return y  

plt.rcParams.update({'font.size':14})


steps =1e-2
t=np.arange(-1,10+steps,steps)

y=step(t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t) with good resolution')
plt.title('Step')



y=ramp(t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,2)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t) with good resolution')
plt.title('Ramp')
plt.xlabel('Time')
plt.show()

steps =1e-2
t=np.arange(-5,10+steps,steps)

def rampstep(t): #defining function for figure 2
    
    return (ramp(t)-ramp(t-3)+5*step(t-3)-2*step(t-6)-2*ramp(t-6))

y= rampstep(t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Step and Ramp Example')
plt.xlabel('Time')
plt.show()

#%% Part 3
#%%%Task 1
steps =1e-2
t=np.arange(-10,5+steps,steps)

y= rampstep(-t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Time Reversal operation')
plt.xlabel('Time')
plt.show()

#%%%Task 2
steps =1e-2
t=np.arange(-14,14+steps,steps)

y1= rampstep(t-4)
y2= rampstep(-t-4)

plt.figure(figsize = (10,7))
plt.plot(t,y1)
plt.plot(t,y2)
plt.grid()
plt.ylabel('y(t)')
plt.title('Time Shift Operation')
plt.xlabel('Time')
plt.show()

#%%%Task 3

steps =1e-2


t=np.arange(-10,20+steps,steps)
y1= rampstep(t/2)


plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y1)
plt.grid()
plt.ylabel('y(t)')
plt.title('Time Scale operation')

t=np.arange(-3,5+steps,steps)
y2= rampstep(2*t)

plt.subplot(2,1,2)
plt.plot(t,y2)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('Time')
plt.show()

#%%%Task 5

steps =1e-5

t=np.arange(-5,10+steps,steps)
y= rampstep(t)
dt = np.diff(t)
dy = np.diff(y, axis = 0)/dt


plt.figure(figsize = (10,7))
plt.plot(t,y, '--', label = 'y(t)')
plt.plot(t[range(len(dy))], dy, label='dy(t)/dt')
plt.grid()
plt.ylabel('y(t)')
plt.title('Derivative Operation')
plt.xlabel('Time')
plt.ylim([-2,10])
plt.show()