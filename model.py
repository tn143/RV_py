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
home=expanduser('~')

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
params=[0,100,0,0.0,0,10]
time=np.arange(0,15,0.001)
rv=rv_pl(time,params)

t=np.random.uniform(0,15,20)
r=rv_pl(t,params)
er=np.random.normal(5,1,len(r))
r=r+(er*np.random.normal(0,1,len(r)))

plt.errorbar(t,r,yerr=er,fmt='.')
plt.plot(time,rv)
plt.axhline(y=0,ls='--',c='k',lw=1)

plt.arrow(10,0,0,100,head_width=0.3, head_length=10,length_includes_head=True,overhang=0.5)
plt.text(10.1,45,r'$K$',fontsize=32)

plt.xticks(fontsize=18)
plt.yticks(fontsize=20)
plt.ylabel(r'Velocity (ms$^{-1}$)',fontsize=20)
plt.xlabel('Time (days)',fontsize=20)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(home+'/Dropbox/PhD/Year_4/Porto/RV.pdf')
plt.show()






