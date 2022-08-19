#carlos.soto362@gmail.com


#COMPUTES THE EIGENVALUES OF CORR_MATRIX
#https://faculty.math.illinois.edu/~z-furedi/PUBS/furedi_komlos_cca1981_random_eig.pdf
#https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190#pgen-0020190-b021

import numpy as np
from pymatreader import read_mat
from matplotlib import cm
import matplotlib.pyplot as plt
import random

from functions import WignerDistribution, plotEigenvalues

f = open('corr_mat/corr_mat.txt','r')
corr_mat = np.array(f.read().split()).astype('float')
f.close()

f = open('corr_mat/corr_mat_Sh.txt','r')
corr_mat_Sh = np.array(f.read().split()).astype('float')
f.close()	

	
corr_mat_Sh = corr_mat_Sh.reshape(100,26*26)
mean_corr_mat_Sh = np.mean(corr_mat_Sh,axis=0)


plotEigenvalues(mean_corr_mat_Sh,'eigenvaluesSh.pdf','eigenvalues of shuffle data')
corr_mat_eiva , corr_mat_eive, indexing = plotEigenvalues(corr_mat,'eigenvalues.pdf','eigenvalues',cap='True')

########################################################################
#let's compare the eigenvectors asociated with the bigest eigenvalues, with the frequency of letters on the inglish alphabet. 
#https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html

frequency_picture_books = np.array([7.95,1.54,2.07,4.47,11.48,1.52,2.41,5.92,5.65,0.14,1.33,4.35,2.28,6.14,7.97,1.57,0.07,5.15,5.54,8.06,3.01,0.77,2.24,0.12,2.28,0.13])
frequency_handwriting_books = np.array([184.6,47,64.1,63.8,247.4,42,65.7,67.7,146.8,25.7,48.8,120.1,51.2,129.5,153.3,68,23.8,120.5,150.7,145,85.9,43.7,48.5,24.8,66.9,27.9])
frequency_common_words = np.array([8000,1600,3000,4400,12000,2500,1700,6400,8000,400,800,4000,3000,8000,8000,1700,500,6200,8000,9000,3400,1200,2000,400,2000,200])
frequency_letters_vocabulary = np.array([43.31,10.56,23.13,17.25,58.88,9.24,12.59,15.31,38.45,1,5.61,27.98,15.36,33.92,36.51,16.14,1,38.64,29.23,35.43,18.51,5.13,6.57,1.48,9.06,1.39])

frequency_picture_books_reescaled = frequency_picture_books/np.linalg.norm(frequency_picture_books)
frequency_handwriting_books_reescaled = frequency_handwriting_books/np.linalg.norm(frequency_handwriting_books)
frequency_common_words_reescaled = frequency_common_words/np.linalg.norm(frequency_common_words)
frequency_letters_vocabulary_reescaled = frequency_letters_vocabulary/np.linalg.norm(frequency_letters_vocabulary)

random.shuffle(frequency_picture_books_reescaled)
random.shuffle(frequency_handwriting_books_reescaled)
random.shuffle(frequency_common_words_reescaled)
random.shuffle(frequency_letters_vocabulary_reescaled)

corr_mat_eive_reescaled = corr_mat_eive/np.linalg.norm(corr_mat_eive,axis=0)

dot_picture_books = np.zeros(26)
dot_handwriting_books = np.zeros(26)
dot_common_words = np.zeros(26)
dot_letters_vocabulary = np.zeros(26)


for i in range(26):

	dot_picture_books[i] = np.dot(frequency_picture_books_reescaled,corr_mat_eive_reescaled[i])
	dot_handwriting_books[i] = np.dot(frequency_handwriting_books_reescaled,corr_mat_eive_reescaled[i])
	dot_common_words[i] = np.dot(frequency_common_words_reescaled,corr_mat_eive_reescaled[i])
	dot_letters_vocabulary[i] = np.dot(frequency_letters_vocabulary_reescaled,corr_mat_eive_reescaled[i])

plt.plot(indexing,dot_picture_books,'o')
plt.show()
plt.plot(indexing,dot_handwriting_books,'o')
plt.show()
plt.plot(indexing,dot_common_words,'o')
plt.show()
plt.plot(indexing,dot_letters_vocabulary,'o')
plt.show()







