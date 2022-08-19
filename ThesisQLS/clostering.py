#carlos.soto362@gmail.com

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist

import numpy as np
from pymatreader import read_mat
from matplotlib import cm
import matplotlib.pyplot as plt
from math import isnan
import random
from scipy import stats
import matplotlib.colors as cls
from tqdm import tqdm

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from functions import plotEigenvalues,WignerDistribution,plot_histogram, Corr_Mat, shuffleL

data = read_mat('Synesthesia-v1.0/eagleman25-Synesthesia-e06c20b/data/EaglemanColoredAlphabets.mat')
colors = np.delete(np.array(data['u_rlab']),0,1)

na=0
nab = np.ones(len(colors)).astype(bool)
for person,i in zip(colors,np.arange(len(colors))):
	if isnan(np.sum(person)):
		na += 1
		nab[i]= False		
colors=colors[nab]

colors_reshape = np.reshape(colors,(len(colors),3*26))

#compute the pairwise distances between points in Lab space
distances = pdist(colors_reshape)

#make the aglomerative ward linkage clostering. Outcome is codified. #This was the one that permormed the best.

closters_linkage = linkage(distances,method='ward',metric='euclidean')

#let's see to what closter, each point belong. 

closter_indexing = fcluster(closters_linkage,4,criterion = 'maxclust')

#storing the clostered data
closters = []
for i in range(np.max(closter_indexing)):
	closters.append(colors[closter_indexing==i+1])



frequency_picture_books = np.array([7.95,1.54,2.07,4.47,11.48,1.52,2.41,5.92,5.65,0.14,1.33,4.35,2.28,6.14,7.97,1.57,0.07,5.15,5.54,8.06,3.01,0.77,2.24,0.12,2.28,0.13])
frequency_handwriting_books = np.array([184.6,47,64.1,63.8,247.4,42,65.7,67.7,146.8,25.7,48.8,120.1,51.2,129.5,153.3,68,23.8,120.5,150.7,145,85.9,43.7,48.5,24.8,66.9,27.9])
frequency_common_words = np.array([8000,1600,3000,4400,12000,2500,1700,6400,8000,400,800,4000,3000,8000,8000,1700,500,6200,8000,9000,3400,1200,2000,400,2000,200])
frequency_letters_vocabulary = np.array([43.31,10.56,23.13,17.25,58.88,9.24,12.59,15.31,38.45,1,5.61,27.98,15.36,33.92,36.51,16.14,1,38.64,29.23,35.43,18.51,5.13,6.57,1.48,9.06,1.39])

frequency_picture_books_reescaled = frequency_picture_books/np.linalg.norm(frequency_picture_books)
frequency_handwriting_books_reescaled = frequency_handwriting_books/np.linalg.norm(frequency_handwriting_books)
frequency_common_words_reescaled = frequency_common_words/np.linalg.norm(frequency_common_words)
frequency_letters_vocabulary_reescaled = frequency_letters_vocabulary/np.linalg.norm(frequency_letters_vocabulary)





for j in tqdm(range(np.max(closter_indexing))):
	N = len(closters[j])
	print(N)
	corr_mat = Corr_Mat(closters[j].reshape((N,3,26)).T)
	corr_mat_Sh = np.zeros(26*26*10)
	for i in range(10):
		colorsSh = shuffleL(closters[j])
		#print('computing covariance matrix of shuffle data {}...'.format(i))
		corr_mat_Sh[26*26*i:26*26*(i+1)] = Corr_Mat(colorsSh.T)
	plot_histogram(corr_mat,60,'blue',label = 'original data')
	plot_histogram(corr_mat_Sh,60,'green',label='shuffle data')
	test = stats.kstest(corr_mat,corr_mat_Sh)
	pV = test[1]
	plt.text(0.8,-plt.ylim()[1]/10,'p_value = %.2E'%pV)
	plt.legend()
	plt.savefig('clostersGraphs/corr_histVrsShuffle_closter{}.pdf'.format(j+1))
	plt.close()
	
	corr_mat_Sh = corr_mat_Sh.reshape(10,26*26)
	mean_corr_mat_Sh = np.mean(corr_mat_Sh,axis=0)


	plotEigenvalues(mean_corr_mat_Sh,'clostersGraphs/eigenvaluesSh_closter{}.pdf'.format(j+1),'eigenvalues of shuffle data')
	corr_mat_eiva , corr_mat_eive, index = plotEigenvalues(corr_mat,'clostersGraphs/eigenvalues_closter{}.pdf'.format(j+1),'eigenvalues',cap='True')
	corr_mat_eive_reescaled = corr_mat_eive/np.linalg.norm(corr_mat_eive,axis=0)
	
	print(np.dot(frequency_picture_books_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))
	print(np.dot(frequency_handwriting_books_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))
	print(np.dot(frequency_common_words_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))
	print(np.dot(frequency_letters_vocabulary_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))
	





