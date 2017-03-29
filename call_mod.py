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

'''
from ajplanet import pl_rv_array as rv_curve
def get_rv(params,time):
	rvsys, K, w, ecc, T0, period=params
	model=rv_curve(time,rvsys, K, np.deg2rad(w), ecc, T0, period)
	return model
'''

home=expanduser('~')
np.random.seed(143)

tlen=5#years
tobs=100#obs

time=np.random.uniform(0,365*tlen,tobs)
time=np.sort(time)
tm=np.linspace(min(time),max(time),100*tobs)
#rvsys, K, w, ecc, T, period=params
rvsys=np.random.uniform(-1e4,1e4)
period=np.random.uniform(0.5,1000)
K=np.random.uniform(1,100)
w=np.random.uniform(0,360)
ecc=np.random.beta(0.867,3.03)
T=np.random.uniform(0,period)
params=[rvsys, K, w, ecc, T, period]
rv=RV(time,params)
rvm=RV(tm,params)

plt.plot(time,rv,'.')
plt.plot(tm,rvm,'--')
plt.show()



















