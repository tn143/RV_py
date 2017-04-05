#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division,print_function
import numpy as np
import sys
import os
from os.path import expanduser
import matplotlib.pyplot as plt
import pandas as pd
#from ajplanet import pl_rv_array as rv_curve
#import gatspy.periodic as gp
from scipy.optimize import fsolve
from scipy import optimize

class RV:
	'First Python Class'
	def __init__(self,time,rv,erv):
		self.time=time
		self.rv=rv
		self.erv=erv
