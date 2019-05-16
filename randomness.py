from math import sqrt
import math
import numpy as np

class randnr:
	def __init__(self, seed=1):
		self.seed = seed
		self.a = 1664525
		self.c = 1013904223
		self.m = 2**32
        
	def randint(self):
		self.seed = (self.a*self.seed+self.c)%self.m
		return self.seed
        
	def random(self):
		return float(self.randint())/float(self.m)
    
	def exp(self,c):
		return -c*math.log(1-self.random())
   
	def gauss(self, mean=0, std=1, m=10):
		x_ls = [self.random() for i in range(m)]
		x = (sum(x_ls)-float(m/2))/sqrt(float(m)/float(12))
		return x*std + mean

### histograms ###

import matplotlib.pyplot as plt
c = randnr(1)

l1 = [c.gauss(0,1,1) for i in range(1000)]
l2 = [c.gauss(0,1,2) for i in range(1000)]
l5 = [c.gauss(0,1,5) for i in range(1000)]
l10 = [c.gauss() for i in range(1000)]

n_bins = 30
fig, ax = plt.subplots(1,4,tight_layout=True,)

