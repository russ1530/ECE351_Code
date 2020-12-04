# ###############################################################
# #
# Kevin Russell #
# ECE 351-51 #
# Lab 12 #
# December 8, 2020 #
# #
# #
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fftpack import fft, fftshift
import pandas as pd
import control as con

def myFFT(x,fs):
    
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

def make_stem ( ax ,x ,y , color ='k', style ='solid', label ='', linewidths =2.5 ,** kwargs ) :
                ax . axhline ( x [0] , x [ -1] ,0 , color ='r')
                ax . vlines (x , 0 ,y , color = color , linestyles = style , label = label , linewidths =
                                     linewidths )
                ax . set_ylim ([1.05* y . min () , 1.05* y . max () ])


#%% load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

plt.figure(figsize = (10,7))
plt.plot(t,sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time[s]')
plt.ylabel('Amplitude [V]')
plt.show()

fs = len(t)/ t[len(t)-1]



X_mag, X_phi, freq = myFFT(sensor_sig,fs)

#%% Task 1

fig, ax1 = plt.subplots(figsize=(10,7))

plt.subplot(ax1)
make_stem(ax1,freq,X_mag)
plt.xscale('log')
plt.xlim([1e0,100e3])
plt.ylim([0,1.5])
plt.grid(which='both')
plt.ylabel('|x(f)|[V]')
plt.title('Spectrum of Noisy Signal (Unfiltered)')
plt.xlabel('Frequency (Hz)')
plt.show()



fig, (ax1, ax2) = plt.subplots(2,1, figsize = (10,7))

plt.subplot(ax1)
make_stem(ax1, freq, X_mag)
plt.xlim([1e0,1.8e3])
plt.xticks([0,60,200,400,600,800,1000,1200,1400,1600,1800])
plt.grid()
plt.title('Spectrum of Noisy Signals in Low and High Bands (Unfiltered)')
plt.ylabel('|x(f)| [V]')

plt.subplot(ax2)
make_stem(ax2, freq, X_mag)
plt.xlim([2e3,10e4])
plt.xticks([2000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000])
plt.grid()
plt.ylabel('|x(f)|[V]')
plt.xlabel('Frequency (Hz)')
plt.show()


fig, ax1 = plt.subplots(figsize=(10,7))


plt.subplot(ax1)
make_stem(ax1,freq,X_mag)
#plt.xscale('log')
plt.xlim([1.79e3,2.01e3])
plt.ylim([0,1.5])
plt.grid()
plt.ylabel('|x(f)|[V]')
plt.title('Spectrum of Desired Signal (Unfiltered)')
plt.xlabel('Frequency (Hz)')
plt.show()

#%%Task 3

steps = 1
omega = np.arange(10**0,10**5+steps,steps)

Hsnum = [4*1256,0]
Hsden = [1,4*1256,(11938**2)]

sys = con.TransferFunction(Hsnum, Hsden)

plt.figure(figsize = (15,10))
_ = con.bode(sys,2*np.pi*omega, dB = True, Hz = True, deg = True, Plot = True)
plt.title('Bode Plot of Filter Circuit')

plt.figure(figsize = (15,10))
_ = con.bode(sys,2*np.pi*np.arange(1.79e3,2.01e3), dB = True, Hz = True, deg = True, Plot = True)
plt.title('Bode Plot of Filter Circuit: Desired Spectrum')


plt.figure(figsize = (15,10))
_ = con.bode(sys,2*np.pi*np.arange(4.5e4,5.5e4), dB = True, Hz = True, deg = True, Plot = True)
plt.title('Bode Plot of Filter Circuit: Switching Amplifier Noise Region')


plt.figure(figsize = (15,10))
_ = con.bode(sys,2*np.pi*np.arange(50,70), dB = True, Hz = True, deg = True, Plot = True)
plt.title('Bode Plot of Filter Circuit: Low-Freqeuncy Vibration Region')



#%% Task 4

z, p = sig.bilinear(Hsnum, Hsden,fs) #converts to z domain from s domain

y_t = sig.lfilter(z,p,sensor_sig)

plt.figure(figsize = (10,7))
plt.plot(t, y_t)
plt.grid()
plt.ylabel('Amplitude [V]')
plt.title('Filtered Output Signal')
plt.xlabel('Time[s]')
plt.show()


X_mag_filtered, X_phi_filtered, freq_filtered = myFFT(y_t,fs)


fig, ax1 = plt.subplots(figsize=(10,7))

plt.subplot(ax1)
make_stem(ax1,freq,X_mag, label = 'Unfiltered Signal')
make_stem(ax1, freq_filtered, X_mag_filtered, color = 'c', style = 'dashed', label = 'Filtered Signal' )
plt.xscale('log')
plt.xlim([1e0,100e3])
plt.ylim([0,1.5])
plt.grid(which='both')
plt.ylabel('|x(f)|[V]')
plt.title('Spectrum of Noisy Signal (Comparison)')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.show()


fig, (ax1, ax2) = plt.subplots(2,1, figsize = (10,7))

plt.subplot(ax1)
make_stem(ax1, freq, X_mag, label = 'Unfiltered Signal')
make_stem(ax1, freq_filtered, X_mag_filtered, color = 'c', style = 'dashed', label = 'Filtered Signal' )
plt.xlim([1e0,1.8e3])
plt.xticks([0,60,200,400,600,800,1000,1200,1400,1600,1800])
plt.grid()
plt.legend()
plt.title('Spectrum of Noisy Signals in Low and High Bands (Comparison)')
plt.ylabel('|x(f)| [V]')

plt.subplot(ax2)
make_stem(ax2, freq, X_mag, label = 'Unfiltered Signal')
make_stem(ax2, freq_filtered, X_mag_filtered, color = 'c', style = 'dashed', label = 'Filtered Signal' )
plt.xlim([2e3,10e4])
plt.xticks([2000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000])
plt.grid()
plt.legend()
plt.ylabel('|x(f)|[V]')
plt.xlabel('Frequency (Hz)')
plt.show()


fig, ax1 = plt.subplots(figsize=(10,7))


plt.subplot(ax1)
make_stem(ax1,freq,X_mag, label = 'Unfiltered Signal')
make_stem(ax1, freq_filtered, X_mag_filtered, color = 'c', style = 'dashed', label = 'Filtered Signal' )
#plt.xscale('log')
plt.xlim([1.79e3,2.01e3])
plt.ylim([0,1.5])
plt.grid()
plt.legend()
plt.ylabel('|x(f)|[V]')
plt.title('Spectrum of Desired Signal (Comparison)')
plt.xlabel('Frequency (Hz)')
plt.show()