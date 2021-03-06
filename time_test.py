from __future__ import division,print_function
import sys
import os
from os.path import expanduser
import matplotlib.pyplot as plt
import pandas as pd
from ajplanet import pl_rv_array as rv_curve
import gatspy.periodic as gp
from scipy.optimize import fsolve
from scipy import optimize
import numpy as np
import time as clock

#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))

def rv_pl(time,params):
	rvsys, K, w, ecc, T, period=params
	w=np.radians(w)
	n=(2*np.pi)/period
	M=n*(time-T)
	E=np.zeros(len(M))
	if ecc==0:
		V=rvsys+K*(np.cos(w+M))
	else:
		if len(time)<150:
			E= fsolve(lambda x: x-ecc*np.sin(x) - M,M)#slower for N >~230
		else:
			for ii,element in enumerate(M): # compute eccentric anomaly
				E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)

		f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))
		V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V



labels=['rvsys', 'K', 'w', 'ecc', 'T', 'period']
params=[10.1,15.3,87.5,0.92,1.2,15.1]

test=np.arange(2,200,1)
for i in test:
	print(i)
	time=np.arange(0,i,0.1)
	s1=clock.clock()
	rv=rv_pl(time,params)
	f1=clock.clock()
	plt.plot(time,rv)
plt.show()







