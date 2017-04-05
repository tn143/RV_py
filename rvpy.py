#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division,print_function
from scipy.optimize import fsolve
import numpy as np

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

	for ii,element in enumerate(M): # compute eccentric anomaly
		E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)
	#E= fsolve(lambda x: x-ecc*np.sin(x) - M,M)#slower for N >~230

	f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))

	V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V





