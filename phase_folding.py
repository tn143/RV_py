#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import corner
from os.path import expanduser
import sys
from scipy.optimize import fsolve
import pandas as pd
import gatspy.periodic as gp
from tqdm import tqdm
from time import sleep
from scipy import stats

home=expanduser('~')

def phase(time, period, origo=0.0, shift=0.5):
    return (((time - origo)/period + shift) % 1.)-0.5

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
	

def fold(time, period, origo=0.0, shift=0.5):
    return (((time - origo)/period + shift) % 1.)-0.5

#time,rv,erv=np.loadtxt('./koi6194.txt',unpack=True,usecols=(0,1,2),skiprows=1)#keck JD - 2440000
#time=time+2440000
#time2,rv2,erv2=np.loadtxt('./K06194.ccfSum.txt',unpack=True,usecols=(0,1,2),skiprows=1)#tres-BJD_UTC 

#np.random.seed(14392)#meaning of life you know?!
#Synthetic
Nobs=30

time=np.random.uniform(0,100,int(Nobs))
time=np.sort(time)
#rvsys, K, w, ecc, T, period
initial=[5,45,100.8956,0.15,13.3345,42.795446754]
rvsys, K, w, ecc, T, period=initial
rv=rv_pl(time,initial)#generate rv curve
erv=2+0.2*np.random.rand(len(time))#random errors ~10ms 
rv+=erv*np.random.randn(len(time))
rv+=np.random.normal(0,1,len(rv))#and random scatter of same level jitter term

plt.errorbar(time,rv,yerr=erv,fmt='.')
plt.show()

mod_time=np.linspace(min(time),max(time),1e4)
model=rv_pl(mod_time,initial)

pha=fold(time,42.795446754,13.3345,0.0)+0.5
phase_mod=fold(mod_time,42.795446754,13.3345,0.0)+0.5
idx=np.argsort(phase_mod)

plt.errorbar(pha,rv,yerr=erv,fmt='.')
#plt.plot(phase_mod[idx],model[idx],'r')
plt.show()
