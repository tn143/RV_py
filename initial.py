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

def get_rv(params,time):
	rvsys, K, w, ecc, T0, period=params
	model=rv_curve(time,rvsys, K, np.deg2rad(w), ecc, T0, period)
	return model


def lomb(time,flux,Nf=15):
	print(type(time))
	if type(time) is np.ndarray:
		pass
	else:
		time=time.values
		flux=flux.values
	time=time-time[0]

	if time[1]<60:
		time=time*86400

	c=[]
	for i in range(len(time)-1):
		c.append(time[i+1]-time[i])
	c=np.median(c)
	nyq=1/(2*(time[1]-time[0]))
	nyq=1/(2*c)
	df=1/time[-1]

	f,p=gp.lomb_scargle_fast.lomb_scargle_fast(time,flux,f0=0,df=df,Nf=Nf*(nyq/df))
	t=1/f#in sec
	return t/86400,p#in days

def phase(time, period, origo=0.0, shift=0.5):
    return (((time - origo)/period + shift) % 1.)-0.5


#rvsys, K, w, ecc, T0, period=initial
#rv_mod=get_rv(initial,time)

#Time	RV		eRV
#BJD	m/s		m/s
#rv=pd.read_csv('KOI-3890_rv.txt')
#rv['Time']-=rv['Time'][0]
time=np.arange(0,10,1)
initial=[10,100,45,0.2,10,1]
rv_mod=get_rv(initial,time)

idx=np.random.randint(0,len(time),10)
tobs=time[[idx]]
rv=rv_mod[idx]
rv+=+np.random.normal(0,25,len(rv))
erv=(0.2*rv) +np.random.normal(0,10,len(rv))
#plt.plot(time,rv_mod)
#plt.errorbar(tobs,rv,erv,fmt='.')
#plt.show()














