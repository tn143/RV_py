#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
import sys
from scipy.optimize import fsolve
import pandas as pd
from tqdm import tqdm
from time import sleep
from rvpy import rv_pl as RV#(time,params)#rvsys, K, w, ecc, T, period=params
from gatspy.periodic import LombScargleFast
from gatspy.periodic import LombScargle

'''
from ajplanet import pl_rv_array as rv_curve
def get_rv(params,time):
	rvsys, K, w, ecc, T0, period=params
	model=rv_curve(time,rvsys, K, np.deg2rad(w), ecc, T0, period)
	return model
'''

home=expanduser('~')
labels=['rvsys', 'K', 'w', 'ecc', 'T', 'period']
#np.random.seed(100)

tlen=0.25#years
tobs=15#obs
year=365.25#days/year

time=np.random.uniform(0,year*tlen,tobs)
time=np.sort(time)
tm=np.linspace(0,max(time),100*tobs)
#rvsys, K, w, ecc, T, period=params
rvsys=np.random.uniform(-1e4,1e4)
period=np.random.uniform(0.5,100)
K=np.random.uniform(1,100)
w=np.random.uniform(0,360)
ecc=np.random.beta(0.867,3.03)
T=np.random.uniform(0,period)
params=[rvsys, K, w, ecc, T, period]

params=[999.9,45.45,w,ecc,np.random.uniform(0,42.3),42.3]
rv=RV(time,params)

erv=np.random.normal(10,3,len(rv))
#erv+=np.random.normal(0,1,len(rv))#added scatter

rv+=np.random.normal(0,np.abs(erv),len(rv))#actually scatter rv points

rvm=RV(tm,params)

print(np.c_[labels,params])

plt.errorbar(time/year,rv,erv,fmt='.')
plt.plot(tm/year,rvm,'--')

plt.show()
#np.savetxt('RV.txt',np.c_[time,rv,erv])
#time,rv,erv=np.loadtxt('hd212771.txt',skiprows=2,comments='#',unpack=True)
#time,rv,erv=np.loadtxt('hd206610.txt',skiprows=2,comments='#',unpack=True)
#time,rv,erv=np.loadtxt('KOI-3890_rv.txt',skiprows=1,delimiter=',',unpack=True)
time,rv,erv=np.loadtxt('kep432.txt',skiprows=0,unpack=True,usecols=(0,1,2))
plt.errorbar(time,rv,erv,fmt='.')
print('Tobs',len(time),'Tlen (yrs)',(time[-1]-time[0])/365.25,'Tlen (days)',(time[-1]-time[0]))
print('mean err ms',np.mean(erv),'err spread ms',np.std(erv))
plt.show()


#LOMB#
pmin=0.1
pmax=max(time)

fmin = 1. / pmax
fmax = 1. / pmin
N = 10000#oversample factor (lots!)
df = (fmax - fmin) / N

#rv=rv-np.nanmean(rv)
model = LombScargle().fit(time, rv)#, erv)
power = model.score_frequency_grid(fmin, df, N)
freqs = fmin + df * np.arange(N)
periods=1./freqs

# plot the results
plt.plot(periods, power)
#plt.xscale('log')
plt.show()

















