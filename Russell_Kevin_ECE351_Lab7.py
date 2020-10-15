# ###############################################################
# #
# Kevin Russell #
# ECE 351-51 #
# Lab 7 #
# October 13, 2020 #
# #
# #
# ###############################################################

#%% Part 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%%%Question 2

numG= [1,9]
denG= sig.convolve([1,-6,-16], [1,4])

numA= [1,4]
denA= [1,4,3]

numB= [1,26,168]
denB= [1]

zG,pG,_ = sig.tf2zpk(numG,denG)
zA,pA,_ = sig.tf2zpk(numA,denA)
zB,pB,_ = sig.tf2zpk(numB,denB)


print ('Zeros for G(s):',zG,'\nPoles for G(s):',pG)
print ('Zeros for A(s):',zA,'\nPoles for A(s):',pA)
print ('Zeros for B(s):',zB,'\nPoles for B(s):',pB)


#%%%Question 5
numYO=sig.convolve(numG,numA)
denYO=sig.convolve(denG,denA)


openStepT, openStepY = sig.step((numYO,denYO))


plt.figure(figsize = (10,20))
plt.plot(openStepT, openStepY)
plt.grid()
plt.ylabel('y(t)')
plt.title('Step response of Open-Loop Transfer Function')


plt.xlabel('Time')
plt.show()


#%%Part 2
#%%%Question 2
numYC = sig.convolve(numA,numG)
denYC = sig.convolve(denA,denG+sig.convolve(numG,numB))

zTot,pTot,_=sig.tf2zpk(numYC,denYC)

print('Closed-Loop Numerator:', numYC, '\nClosed-Loop Denominator:', denYC)
print('Closed-Loop Zeros:', zTot, '\nClosed-Loop poles:', np.round(pTot,2))


#%%%Question 4
closedStepT, closedStepY = sig.step((numYC,denYC))

plt.figure(figsize = (10,20))
plt.plot(closedStepT, closedStepY)
plt.grid()
plt.ylabel('y(t)')
plt.title('Step response of Closed-Loop Transfer Function')
plt.xlabel('Time')
plt.show()

