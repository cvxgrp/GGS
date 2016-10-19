from ggs import *
import numpy as np
import matplotlib.pyplot as plt

#Read in daily log returns from 1997-2015 of 10 indices: 
#DM stocks, EM stocks, Real estate, Oil, Gold, HY bonds, EM HY bonds, GVT bonds, CORP bonds, IFL bonds
filename = "Data/Returns.txt"
data = np.genfromtxt(filename,delimiter=' ')
#Select DM stocks, Oil, and GVT bonds
feats = [0,3,7]




#Find 10 breakpoints at lambda = 1e-4
bps, objectives = RunGGS(data, K = 10, lamb = 1e-4, features = feats)

#Find means and covariances of the segments, given the selected breakpoints
meancovs = FindMeanCovs(data, breakpoints = bps, lamb = 1e-4, features = feats)


print "Breakpoints are at", bps

#Plot objective vs. number of breakpoints
plotVals = map(list, zip(*objectives))
plotVals[0] = [len(i)-2 for i in plotVals[0]]
plt.plot(plotVals[0], plotVals[1], 'or-')
plt.xlabel('Number of Breakpoints')
plt.ylabel(r'$\phi(b)$')
plt.show()
