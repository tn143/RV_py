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
	
def lnprior(theta):
	rvsys, K, w, ecc, Tr, period, sig,rvsys2,sig2 = theta
	logp=np.log10(period)
	logr=np.log10(rvsys)
	logk=np.log10(K)
	sqrte=np.sqrt(ecc)
	logsig=np.log10(sig)
	logsig2=np.log10(sig2)
	#if -100<rvsys<100 and -2 < logk < 3 and 0.0<w<360 and 30<period<50 and 2440000+17950<Tr<2440000+18000:
	#	ln_e=stats.beta(0.867,3.03)
	#	lnp_per=-0.5*(period-42.295)**2/(0.0001**2)
	#	lne=-0.5*(ecc-0.01)**2/(0.0001**2)
	#	return ln_e.logpdf(ecc)+lnp_per+lne
	if -100<rvsys<100 and -100<rvsys2<100 and -2 < logk < 3 and -1<sqrte*np.cos(np.deg2rad(w))<1\
	and -1<sqrte*np.sin(np.deg2rad(w))<1 and ecc<1 and -10<logsig<2 and -10<logsig2<2\
	and 0<Tr<45 and  30<period<44 and 0<w<360:
		lnp_per=-0.5*(period-42.295)**2/(0.0001**2)
		return lnp_per
	else:
		return -np.inf


def lnlike(theta, time, rv, erv,tim2,rv2,erv2):
	rvsys, K, w, ecc, T0, period,sig,rvsys2,sig2 = theta
	p1=[rvsys,K,w,ecc,T0,period]
	p2=[rvsys2,K,w,ecc,T0,period]

	model =rv_pl(time,p1)
	inv_sigma2 = 1.0/(sig**2+erv**2)# + model**2)
	r1=-0.5*(np.sum((rv-model)**2*inv_sigma2 - np.log(inv_sigma2)))

	model =rv_pl(time2,p2)
	inv_sigma2 = 1.0/(sig2**2+erv2**2)# + model**2)
	r2=-0.5*(np.sum((rv2-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	return r1+r2



def lnprob(theta, time, rv, erv,time2,rv2,erv2):
    lp = lnprior(theta)
    if not np.isfinite(lp):
		return -np.inf
    return lp + lnlike(theta, time, rv, erv, time2, rv2, erv2)


def phase(time, period, origo=0.0, shift=0.5):
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

time2=max(time)+np.random.uniform(100,200,Nobs/2)
time2=np.sort(time2)
#rvsys, K, w, ecc, T, period
initial=[10,45,100.8956,0.15,13.3345,42.795446754]
rvsys, K, w, ecc, T, period=initial
rv2=rv_pl(time2,initial)#generate rv curve
erv2=10+0.2*np.random.rand(len(time2))#random errors ~10ms 
rv2+=erv2*np.random.randn(len(time2))
rv2+=np.random.normal(0,2,len(rv2))#and random scatter of same level jitter term

plt.errorbar(time,rv,yerr=erv,fmt='.')
plt.errorbar(time2,rv2,yerr=erv2,fmt='.')
plt.show()

mod_time=np.linspace(min(time),max(time2),1e4)
model=rv_pl(mod_time,initial)

pha=phase(time,13.3345,42.795446754,0.0)+0.5
phase_mod=phase(mod_time,13.3345,42.795446754,0.0)+0.5
idx=np.argsort(phase_mod)

plt.errorbar(pha,rv,yerr=erv,fmt='.')
plt.plot(phase_mod[idx],model[idx],'r')
plt.show()





from gatspy.periodic import LombScargleFast
from gatspy.periodic import LombScargle
N = 10000
periods=np.linspace(10,2*max(time-time[0]),N)
fmin = 1. / periods.max()
fmax = 1. / periods.min()
df = (fmax - fmin) / N

model = LombScargle().fit(time-time[0], rv, erv)
power = model.score_frequency_grid(fmin, df, N)
freqs = fmin + df * np.arange(N)

periods=1./freqs
# plot the results
plt.plot(1. / freqs, power)
plt.show()

#rvsys, K, w, ecc, T0, period
labels=['rvsys', 'K', 'w', 'ecc', 'T0', 'period','sig','rvsys2','sig2']
initial=[0,40,180,0.1,20,42.5,10,0,10]

#plt.errorbar(time,rv,erv,fmt='.')
#plt.plot(mod_time,rv_pl(mod_time,initial))
#plt.show()

# Set up the sampler.
ntemps, nwalkers, niter, ndim = 2, 300, 800, len(labels)
sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprior, loglargs=(time, rv, erv,time2,rv2,erv2))
p0 = np.zeros([ntemps, nwalkers, ndim])
for i in range(ntemps):
    for j in range(nwalkers):
	p0[i,j,:] = initial + 1e-2*np.random.randn(ndim)

print('... burning in ...')
for p, lnprob, lnlike in tqdm(sampler.sample(p0, iterations=niter),total=niter):
	sleep(0.001)

#print(np.shape(p0))
#print(np.shape(p))
# Clear and run the production chain.
sampler.reset()

print('... running sampler ...')
for p, lnprob, lnlike in tqdm(sampler.sample(p, lnprob0=lnprob,lnlike0=lnlike,iterations=niter),total=niter):
	sleep(0.001)

fig, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(8, 9))

for i in range(0,len(initial)):
	axes[i].plot(sampler.chain[0, :, :, i].T, color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylabel(labels[i])

fig.tight_layout(h_pad=0.0)
plt.show()
#print('n temps:', ntemps, "log evidence: ", sampler.thermodynamic_integration_log_evidence())

# Make the corner plot.
burnin = (niter/2)
samples = sampler.chain[0, :, burnin:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=labels)
plt.show()

quantiles = np.percentile(samples,[16,50,84],axis=0).T 
medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-

np.savetxt('RV_results.txt',np.c_[np.array(labels),medians,uerr,lerr],fmt='%s',header='Param,median,uerr,lerr')

for i in range(0,len(labels)):
	print(labels[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))

rvsys, K, w, ecc, T0, period=medians
plt.errorbar(time,rv,erv,fmt='.')
plt.plot(mod_time,rv_pl(mod_time,medians))
plt.show()


