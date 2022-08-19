#carlos.soto362@gmail.com


#CREATES THE HISTOGRAM OF VALUES OF THE CORR_MATRIX USING THE NORMAL DEFINITION.


import numpy as np
from pymatreader import read_mat
from matplotlib import cm
import matplotlib.pyplot as plt
from math import isnan
import random
from scipy import stats
import matplotlib.colors as cls

from functions import Corr_Mat_People, Corr_Mat, plot_histogram, mean, var, corr


data = read_mat('Synesthesia-v1.0/eagleman25-Synesthesia-e06c20b/data/EaglemanColoredAlphabets.mat')
colors = np.delete(np.array(data['u_rlab']),0,1)

#getting rid of elements with nan

na=0
nab = np.ones(len(colors)).astype(bool)
for person,i in zip(colors,np.arange(len(colors))):
	if isnan(np.sum(person)):
		na += 1
		nab[i]= False		
colors=colors[nab]

#calculating the distribution of values for the covariance matrix
#######################################################

			
'''
print('calculating the covariance matrix for {} data points over people...'.format(len(colors)))
corr_mat_p = Corr_Mat_People(colors)
f = open('corr_mat_p.txt','w')
for i in corr_mat_p:
	f.write(str(i) + ' ')
f.close()
'''
'''
print('calculating the covariance matrix for {} data points...'.format(len(colors)))
corr_mat = Corr_Mat(colors.T)

f = open('corr_mat.txt','w')
for i in corr_mat:
	f.write(str(i) + ' ')
f.close()
'''
###########################################################
#printing the distribution of values for the covariance matrix	
#reshuffling and printing the histogram of the random correlation matrix. The shuffle is done over the colors of a same letter. 
#########################################
'''
random.seed(2022)

def shuffleL(colors_):
	colorsSh = []

	for i in range(26):
		listS = list(colors_.T[i].T)
		random.shuffle(listS)
		colorsSh.append(np.array(listS).T)
	return np.array(colorsSh).T


corr_mat_Sh = np.zeros(26*26*100)
for i in range(100):
	colorsSh = shuffleL(colors)
	print('computing covariance matrix of shuffle data {}...'.format(i))
	corr_mat_Sh[26*26*i:26*26*(i+1)] = Corr_Mat(colorsSh.T)
corr_mat_Sh = corr_mat_Sh

f = open('corr_mat_Sh.txt','w')
for i in corr_mat_Sh:
	f.write(str(i) + ' ')
f.close()
'''
############################################
f = open('corr_mat/corr_mat.txt','r')
corr_mat = np.array(f.read().split()).astype('float')
f.close()
#printing the distribution of values for the covariance matrix

plot_histogram(corr_mat,60,'blue',label = 'original data')

f = open('corr_mat/corr_mat_Sh.txt','r')
corr_mat_Sh = np.array(f.read().split()).astype('float')
f.close()

plot_histogram(corr_mat_Sh,60,'green',label='shuffle data')

test = stats.kstest(corr_mat,corr_mat_Sh)
pV = test[1]
plt.text(0.8,-0.03,'p_value = %.2E'%pV)

plt.legend()
plt.savefig('corr_histVrsShuffle.pdf')
plt.close()


corr_mat_Sh = corr_mat_Sh.reshape(100,26*26)
mean_corr_mat_Sh = np.mean(corr_mat_Sh,axis=0)
corr_mat_Sh = np.reshape(mean_corr_mat_Sh,(26,26))
print(corr_mat_Sh[0,0])
	

cmap = cm.get_cmap('plasma', 256)

psm = plt.pcolormesh(corr_mat_Sh, cmap=cmap, rasterized=True, norm=cls.SymLogNorm(linthresh =0.001,linscale=0.4  ,vmin=corr_mat_Sh.min(), vmax=corr_mat_Sh.max()))
plt.colorbar(psm)


plt.savefig('corr_mat/corr_mat_Sh.pdf')
plt.close()


print(len(colors))









