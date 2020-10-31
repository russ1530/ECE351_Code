# ###############################################################
# #
# Kevin Russell #
# ECE 351-51 #
# Lab 8 #
# October 20, 2020 #
# #
# #
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift



def myFFT(x,fs):
    
    N= len(x)
    X_fft = fft(x)
    X_fft_shifted = fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
     
    
    return X_mag, X_phi, freq


def myFFT_CLEAN(x,fs):
    
    N= len(x)
    X_fft = fft(x)
    X_fft_shifted = fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    #X_phi_clean = np.zeros
    
    for i in range(len(X_mag)):
        if np.abs(X_mag[i])< 1e-10:
            X_phi[i]=0
        else:
            X_phi[i]=X_phi[i]
    
    
    return X_mag, X_phi, freq


#%% Task 1

fs=100


steps =1/fs
t=np.arange(0,2,steps)

x = np.cos(2*np.pi*t)

plt.figure(figsize = (15,12))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t)')
plt.title('Task1 - User-Defined FFT of cosine')
plt.xlabel('Time(s)')


X_mag, X_phi, freq = myFFT(x,fs)


plt.subplot(3,2,3)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.ylabel('|x(f)|')
plt.subplot(3,2,5)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.ylabel('/_x(f)')
plt.xlabel('f(Hz)')

plt.subplot(3,2,4)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.xlim(-2,2)
plt.subplot(3,2,6)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.xlim(-2,2)
plt.xlabel('f(Hz)')

plt.show()


#%%Task 2

fs=100


steps =1/fs
t=np.arange(0,2,steps)

x = 5*np.sin(2*np.pi*t)

plt.figure(figsize = (15,12))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t)')
plt.title('Task1 - User-Defined FFT of sine')
plt.xlabel('Time(s)')


X_mag, X_phi, freq = myFFT(x,fs)


plt.subplot(3,2,3)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.ylabel('|x(f)|')
plt.subplot(3,2,5)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.ylabel('/_x(f)')
plt.xlabel('f(Hz)')

plt.subplot(3,2,4)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.xlim(-2,2)
plt.subplot(3,2,6)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.xlim(-2,2)
plt.xlabel('f(Hz)')

plt.show()

#%%Task 3

fs=100


steps =1/fs
t=np.arange(0,2,steps)

x = 2*np.cos(2*np.pi*2*t)+(np.sin((2*np.pi*6*t)+3))**2

plt.figure(figsize = (15,12))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t)')
plt.title('Task1 - User-Defined FFT of cosine/sine')
plt.xlabel('Time(s)')


X_mag, X_phi, freq = myFFT(x,fs)


plt.subplot(3,2,3)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.ylabel('|x(f)|')
plt.subplot(3,2,5)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.ylabel('/_x(f)')
plt.xlabel('f(Hz)')

plt.subplot(3,2,4)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.xlim(-20,20)
plt.subplot(3,2,6)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.xlim(-20,20)
plt.xlabel('f(Hz)')

plt.show()

#%%Task 4

fs=100


steps =1/fs
t=np.arange(0,2,steps)

x = np.cos(2*np.pi*t)

plt.figure(figsize = (15,12))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t)')
plt.title('Task4 - User-Defined FFT of cosine CLEAN')
plt.xlabel('Time(s)')


X_mag, X_phi, freq = myFFT_CLEAN(x,fs)


plt.subplot(3,2,3)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.ylabel('|x(f)|')
plt.subplot(3,2,5)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.ylabel('/_x(f)')
plt.xlabel('f(Hz)')

plt.subplot(3,2,4)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.xlim(-2,2)
plt.subplot(3,2,6)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.xlim(-2,2)
plt.xlabel('f(Hz)')

plt.show()

fs=100


steps =1/fs
t=np.arange(0,2,steps)

x = 5*np.sin(2*np.pi*t)

plt.figure(figsize = (15,12))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t)')
plt.title('Task4 - User-Defined FFT of sine CLEAN')
plt.xlabel('Time(s)')


X_mag, X_phi, freq = myFFT_CLEAN(x,fs)


plt.subplot(3,2,3)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.ylabel('|x(f)|')
plt.subplot(3,2,5)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.ylabel('/_x(f)')
plt.xlabel('f(Hz)')

plt.subplot(3,2,4)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.xlim(-2,2)
plt.subplot(3,2,6)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.xlim(-2,2)
plt.xlabel('f(Hz)')

plt.show()

fs=100


steps =1/fs
t=np.arange(0,2,steps)

x = 2*np.cos(2*np.pi*2*t)+(np.sin((2*np.pi*6*t)+3))**2

plt.figure(figsize = (15,12))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t)')
plt.title('Task4 - User-Defined FFT of cosine/sine CLEAN')
plt.xlabel('Time(s)')


X_mag, X_phi, freq = myFFT_CLEAN(x,fs)


plt.subplot(3,2,3)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.ylabel('|x(f)|')
plt.subplot(3,2,5)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.ylabel('/_x(f)')
plt.xlabel('f(Hz)')

plt.subplot(3,2,4)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.xlim(-20,20)
plt.subplot(3,2,6)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.xlim(-20,20)
plt.xlabel('f(Hz)')

plt.show()

#%%Task 5

def a(k):
    
   return 0

def b(k):
    
    b=(2-2*np.cos(np.pi*k))/(np.pi*k)
    
    return b

T=8
omega=(2*np.pi)/T

def FS(t,T,k):
    x=0
    omega = (2*np.pi)/T
    for j in range (1,k+1):
        x = x + b(j)*np.sin(j*omega*t)
        
    return x




fs=100


steps =1/fs
t=np.arange(0,16,steps)

x = FS(t,8,15)

plt.figure(figsize = (15,12))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t)')
plt.title('Task5 - User-Defined FFT of Square Wave CLEAN')
plt.xlabel('Time(s)')


X_mag, X_phi, freq = myFFT_CLEAN(x,fs)


plt.subplot(3,2,3)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.ylabel('|x(f)|')
plt.subplot(3,2,5)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.ylabel('/_x(f)')
plt.xlabel('f(Hz)')

plt.subplot(3,2,4)
plt.stem(freq,X_mag,use_line_collection=(True))
plt.xlim(-3,3)
plt.subplot(3,2,6)
plt.stem(freq, X_phi,use_line_collection=(True))
plt.xlim(-3,3)
plt.xlabel('f(Hz)')

plt.show()