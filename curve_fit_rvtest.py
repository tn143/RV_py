from __future__ import division,print_function
import numpy as np
import scipy
import sys
import os
from os.path import expanduser
import matplotlib.pyplot as plt
import pandas as pd
import gatspy.periodic as gp
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))

def rv_pl(time,rvsys, K, w, ecc, T, period):
	w=np.radians(w)
	if w<0:
		w=w+(2*np.pi)
	if ecc<0:
		ecc=np.abs(ecc)

	print(ecc,np.rad2deg(w))
	n=(2*np.pi)/period
	M=n*(time-T)
	E=np.zeros(len(M))
	for ii,element in enumerate(M): # compute eccentric anomaly
		E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)
	f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))

	V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V

#Time	RV		eRV
#BJD	m/s		m/s
#rv=pd.read_csv('KOI-3890_rv.txt')
#rv['Time']-=2400000
#rv['RV']/=1e3
#rv['eRV']/=1e3
#time,rvsys, K, w, ecc, T, period
labels=['rvsys', 'K', 'w', 'ecc', 'T', 'period']

'''
#Synthetic
time=np.random.randint(0,4*365,50)#50 observations over 4 years in days
time=np.sort(time)
mod_time=np.linspace(min(time),max(time),1e4)
p=np.random.randint(0.2,365)#from 0.2days to 1 years
initial=[np.random.randint(10,20),np.random.randint(10,200),np.random.randint(0,360),np.random.random(),p*np.random.random(),p]#make the planet
rvsys, K, w, ecc, T, period=initial
rv=rv_pl(time,rvsys, K, w, ecc, T, period)#generate rv curve
erv=np.random.normal(5,5,len(rv))#random errors ~10ms 
erv+=np.random.normal(0,5,len(rv))#and random scatter of same level
'''

#Real
#rv=pd.read_csv('hd212771.txt',comment='#',delim_whitespace=True)#MIKE
#initial=[1,50,45,0.2,time[rv==min(rv)],365]


#rv=pd.read_csv('KOI-3890_rv.txt')
#rv['Time']-=2400000
#rv['RV']/=1e3
#rv['eRV']/=1e3
#time,rv,erv=rv['Time'].values,rv['RV'].values,rv['eRV'].values

#Real
rv=pd.read_csv('hd206610.txt',comment='#',delim_whitespace=True)#HD 206610
time,rv,erv=rv['Time'].values,rv['RV'].values,rv['eRV'].values



initial=[20,40,270,0.15,time[rv==min(rv)],600]

mod_time=np.linspace(min(time),max(time),1e4)
plt.errorbar(time,rv,erv,fmt='.')
for r in range(0,10):
	print('Run',r)
	try:
		results#if list doesn't exist, make it
	except NameError:
		results=[]


	guess=np.zeros(len(initial))#reasonable initial guess with some scatter
	for i ,v in enumerate(initial):
		err=np.random.normal(0,np.abs(v/4))
		guess[i]=v+err
	
	#print(np.c_[labels,initial,guess])
	try:
		popt, pcov = curve_fit(rv_pl, time, rv,sigma=erv,p0=guess,maxfev=5000)
		chi =np.sum(((rv-rv_pl(time,*popt))/erv)**2 )

		if chi<1000:#keep goodish fits
			print(np.c_[labels,popt,np.sqrt(np.diag(pcov))])
			results.append([r,chi])  
			plt.plot(mod_time,rv_pl(mod_time,*popt),'--',label=r)

		else:
			results.append([r,np.inf])  

	except RuntimeError:#if failed record inf
		results.append([r,np.inf])

results=np.array(results)
print(results)
plt.legend()
plt.show()






