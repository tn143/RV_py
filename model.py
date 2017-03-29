from __future__ import division,print_function
import numpy as np
import scipy
import sys
import os
from os.path import expanduser
import matplotlib.pyplot as plt
import pandas as pd
from ajplanet import pl_rv_array as rv_curve
import gatspy.periodic as gp
from scipy.optimize import fsolve
from scipy import optimize
from ajplanet import pl_true_anomaly as tru
import time as clock


#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))

def get_rv(params,time):
	rvsys, K, w, ecc, T0, period=params
	model=rv_curve(time,rvsys, K, np.deg2rad(w), ecc, T0, period)
	return model

def rv_pl(params,time):
	rvsys, K, w, ecc, T, period=params
	w=np.radians(w)
	n=(2*np.pi)/period
	M=n*(time-T)
	E=np.zeros(len(M))

	for ii,element in enumerate(M): # compute eccentric anomaly
		E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)
	#E= fsolve(lambda x: x-ecc*np.sin(x) - M,M)#slower for N >~230

	f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))

	V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V



for i in range(0,10):

	#np.random.seed(143)
	#Synthetic
	labels=['rvsys', 'K', 'w', 'ecc', 'T', 'period']
	time=np.random.randint(0,4*365,20)#50 observations over 4 years in days
	time=np.sort(time)

	#rvsys, K, w, ecc, T, period
	for i in range(0,3):
		rvsys=0#np.random.uniform(10,20)
		if i>0:
			K=np.random.uniform(10,200)
			ecc=ecc+np.random.uniform(-0.1,0.1)
			w=w+np.random.uniform(-5,5)
			p=p*np.random.uniform(2,20)
			T=np.random.uniform(0,p)
		else:
			K=np.random.uniform(10,200)
			ecc=np.random.beta(0.867,3.03)
			w=np.random.uniform(0,360)
			p=np.random.uniform(0.2,365)#from 0.2days to 1 years
			T=np.random.uniform(0,p)


		mod_time=np.linspace(0,4*365,1000)
		initial=[rvsys,K,w,ecc,T,p]#make the planet
		if i ==0:
			rv=rv_pl(initial,time)#generate rv curve
			rv_mod=rv_pl(initial,mod_time)

		else:
			rv=rv+rv_pl(initial,time)#generate rv curve
			rv_mod=rv_mod+rv_pl(initial,mod_time)

	
		erv=np.random.normal(5,10,len(rv))#random errors ~10ms 
		erv+=np.random.normal(0,10,len(rv))#and random scatter of same level

		#rv_aj=get_rv(initial,mod_time)#generate rv curve
		plt.plot(mod_time,rv_pl(initial,mod_time),'-',lw=1,alpha=0.4)

	plt.errorbar(time,rv,erv,fmt='.',label='Data')
	plt.plot(mod_time,rv_mod,'--',label='Thomas',lw=2,alpha=0.8)
	plt.legend()
	plt.show()






