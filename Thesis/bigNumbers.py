#carlos.soto362@gmail.com
from colormath.color_objects import LabColor,sRGBColor
from colormath.color_conversions import convert_color
import numpy as np
from pymatreader import read_mat
from matplotlib import cm
import matplotlib.pyplot as plt
from math import isnan
import random
import scipy
from scipy import stats
import matplotlib.colors as cls
from tqdm import tqdm

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from functions import plotEigenvalues,WignerDistribution,plot_histogram, Corr_Mat, shuffleL

data = read_mat('Synesthesia-v1.0/eagleman25-Synesthesia-e06c20b/data/EaglemanColoredAlphabets.mat')
colors = np.delete(np.array(data['u_rlab']),0,1)

dataN = read_mat('Synesthesia-v1.0/eagleman25-Synesthesia-e06c20b/data/lRGBnathan.mat')
#print(dataN['labeledRGB'])

na=0
nab = np.ones(len(colors)).astype(bool)
for person,i in zip(colors,np.arange(len(colors))):
	if isnan(np.sum(person)):
		na += 1
		nab[i]= False		
colors=colors[nab]




#################################################################################################
#write the frequency lists
#################################################################################################

frequency_picture_books = np.array([7.95,1.54,2.07,4.47,11.48,1.52,2.41,5.92,5.65,0.14,1.33,4.35,2.28,6.14,7.97,1.57,0.07,5.15,5.54,8.06,3.01,0.77,2.24,0.12,2.28,0.13])
frequency_handwriting_books = np.array([184.6,47,64.1,63.8,247.4,42,65.7,67.7,146.8,25.7,48.8,120.1,51.2,129.5,153.3,68,23.8,120.5,150.7,145,85.9,43.7,48.5,24.8,66.9,27.9])
frequency_common_words = np.array([8000,1600,3000,4400,12000,2500,1700,6400,8000,400,800,4000,3000,8000,8000,1700,500,6200,8000,9000,3400,1200,2000,400,2000,200])
frequency_letters_vocabulary = np.array([43.31,10.56,23.13,17.25,58.88,9.24,12.59,15.31,38.45,1,5.61,27.98,15.36,33.92,36.51,16.14,1,38.64,29.23,35.43,18.51,5.13,6.57,1.48,9.06,1.39])
#[e,t,a,o,i,n,s,r,h,l ,d ,c ,u ,m ,f ,p ,g ,w ,y ,b ,v ,k ,x ,j ,q ,z ]
#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

frequency_BNC = np.array([3,20,12,11,1,15,17,9,5,24,22,10,14,6,4,16,25,8,7,2,13,21,18,23,19,26])

frequency_picture_books_reescaled = frequency_picture_books/np.linalg.norm(frequency_picture_books)
frequency_handwriting_books_reescaled = frequency_handwriting_books/np.linalg.norm(frequency_handwriting_books)
frequency_common_words_reescaled = frequency_common_words/np.linalg.norm(frequency_common_words)
frequency_letters_vocabulary_reescaled = frequency_letters_vocabulary/np.linalg.norm(frequency_letters_vocabulary)
frequency_BNC_reescaled = frequency_BNC/np.linalg.norm(frequency_BNC)

#let's plot the frequencies with the luminosity, hue and 

str_colors = ['a','b','c','d','e','f','g','h', 'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

##############################################################################################
#definding the magnet toy
##############################################################################################
red = np.array(convert_color(sRGBColor(*np.array([1, 0, 0])), LabColor).get_value_tuple())
orange = np.array(convert_color(sRGBColor(*np.array([1, 0.5, 0])), LabColor).get_value_tuple())
yellow = np.array(convert_color(sRGBColor(*np.array([1, 1, 0])), LabColor).get_value_tuple())
green = np.array(convert_color(sRGBColor(*np.array([0, 1, 0])), LabColor).get_value_tuple())
blue = np.array(convert_color(sRGBColor(*np.array([0, 0, 1])), LabColor).get_value_tuple())
purple = np.array(convert_color(sRGBColor(*np.array([0.5, 0, 1])), LabColor).get_value_tuple())
black = np.array(convert_color(sRGBColor(*np.array([0, 0, 0])), LabColor).get_value_tuple())


magnets = np.array([red,orange,yellow,green,blue,purple,red,orange,yellow,green,blue,purple,red,orange,yellow,green,blue,purple,red,orange,yellow,green,blue,purple,red,orange])
corr_matrix_magnets = np.corrcoef(magnets)
eigen_magnets = scipy.linalg.eigh(corr_matrix_magnets)

#################################################################################################
#before shoffling, let's try a diferent messure of correlation. 
#################################################################################################


#for each point, lets compute the dot product between the chromas of each letter and the frequency of letters in color the picture books. 


colors_luminosity = colors[:,0]
colors_hue_chroma = colors[:,1:3]
colors_chroma = np.sqrt(colors_hue_chroma[:,0]**2 + colors_hue_chroma[:,1]**2)
colors_hue_chroma[colors_hue_chroma == 0] = 0.000001
colors_hue = np.array(np.arctan(colors_hue_chroma[:,1]/colors_hue_chroma[:,0]))

print(np.min(colors_luminosity))
print(np.min(colors_hue))
print(np.min(colors_chroma))


def independent_correlation(colors_,frequency):
	dot=0
	for person in colors_:
		person_reescaled = person/np.linalg.norm(person)
		dot += np.dot(person_reescaled,frequency)
	dot = dot/len(colors_)
	return dot
dot= independent_correlation(colors_chroma,frequency_picture_books_reescaled)
print('mean value of dot: ',dot)

#################################################################################################
#lets make the dot product with random frequencys and see if it looks a rando frequency option
#################################################################################################
#Dot_random_frequencys = []
#for i in tqdm(range(1000)):
#	frequencys = np.random.random(26)
#	frequencys_reescaled = frequencys/np.linalg.norm(frequencys)
#	Dot_random_frequencys.append(independent_correlation(colors_chroma,frequencys_reescaled))
#plt.hist(Dot_random_frequencys)
#plt.show()
#plt.close()

#it looks like 0.67 is into the normal range, aprox [0.6-0.8]

#################################################################################################
#lets try if clusters have big eigenvec aligned with frequencys, starting with chroma,closter 1.
#################################################################################################
'''#0.4 seems to be the limit. 
Dot_chroma = []
for i in tqdm(range(1000)):
	colors_ = shuffleL(colors)
	colors_ = colors_[:690]
	colors_hue_chroma = colors_[:,1:3]
	colors_chroma = np.sqrt(colors_hue_chroma[:,0]**2 + colors_hue_chroma[:,1]**2)
	
	#compute the correlation matrix

	corr_mat = Corr_Mat(colors_chroma)

	#computing the eigenvalues of chroma
	corr_mat = np.reshape(corr_mat,(26,26))
	eigen = np.linalg.eig(corr_mat)
	eigenvalues = eigen[0]
	eigenvectors = eigen[1]
	corr_mat_eive_reescaled = eigenvectors/np.linalg.norm(eigenvectors,axis=0)

	#compute the dot product with picture books (bigest dot product in non shuffle data)
	Dot_chroma.append(np.dot(frequency_picture_books_reescaled,corr_mat_eive_reescaled[np.argmax(eigenvalues)]))
	
weights = np.ones_like(np.array(Dot_chroma))/float(len(Dot_chroma))

test = stats.kstest((np.array(Dot_chroma) - np.mean(np.array(Dot_chroma)) )/np.std(np.array(Dot_chroma))  ,'norm')
pV = test[1]
plt.hist( np.array(Dot_chroma)   ,bins=20,label='mean = {}\n std = {}\np_value = {}'.format(np.mean(np.array(Dot_chroma)),np.std(np.array(Dot_chroma)),pV),weights=weights)

#plt.plot(np.linspace(-1,1,100),0.05*stats.norm.pdf(np.linspace(-1,1,100),scale=np.std(np.array(Dot_hue))))
plt.xlabel('dot product between the maximun eigenvalue \n of the shuffle data and the frequency of letters in picture books')
plt.ylabel('frequency')
plt.legend()
plt.show()
#plt.savefig('hue/dotPicturesMaxEigShoffleHist.pdf')
plt.close()
'''
#################################################################################################
#0.4 seems to be the limit. lets try with the toys.
#################################################################################################
'''
Dot_chroma = []
for i in tqdm(range(1000)):
	colors_ = shuffleL(colors)
	colors_ = colors_[:751]
	colors_hue_chroma = colors_[:,1:3]
	colors_chroma = np.sqrt(colors_hue_chroma[:,0]**2 + colors_hue_chroma[:,1]**2)
	
	#compute the correlation matrix

	corr_mat = Corr_Mat(colors_chroma)

	#computing the eigenvalues of chroma
	corr_mat = np.reshape(corr_mat,(26,26))
	eigen = np.linalg.eig(corr_mat)
	eigenvalues = eigen[0]
	eigenvectors = eigen[1]
	corr_mat_eive_reescaled = eigenvectors/np.linalg.norm(eigenvectors,axis=0)

	#compute the dot product with picture books (bigest dot product in non shuffle data)
	Dot_chroma.append(np.dot(eigen_magnets[1][np.argmax(eigen_magnets[0])],corr_mat_eive_reescaled[np.argmax(eigenvalues)]))
	
weights = np.ones_like(np.array(Dot_chroma))/float(len(Dot_chroma))

test = stats.kstest((np.array(Dot_chroma) - np.mean(np.array(Dot_chroma)) )/np.std(np.array(Dot_chroma))  ,'norm')
pV = test[1]
plt.hist( np.array(Dot_chroma)   ,bins=20,label='mean = {}\n std = {}\np_value = {}'.format(np.mean(np.array(Dot_chroma)),np.std(np.array(Dot_chroma)),pV),weights=weights)

#plt.plot(np.linspace(-1,1,100),0.05*stats.norm.pdf(np.linspace(-1,1,100),scale=np.std(np.array(Dot_hue))))
plt.xlabel('dot product between the maximun eigenvalue \n of the shuffle data and the frequency of letters in picture books',fontzise=10)
plt.ylabel('frequency')
plt.legend()
plt.show()
#plt.savefig('hue/dotPicturesMaxEigShoffleHist.pdf')
plt.close()
'''
#################################################################################################
#shuffle the data many times, and see the maximun value obtained for the correlations
#################################################################################################



Dot_chroma = []
Dot_hue = []
Dot_luminosity = []
#ind_corr_chroma = []      #0.6 is not a rare event. 0.67 is a rare event in comparison with 0.66, wich is the normal range, but, I don't know how much...
for i in tqdm(range(1000)):
	colors_ = shuffleL(colors)

	#define the chroma

	colors_luminosity = colors_[:,0]
	colors_hue_chroma = colors_[:,1:3]
	colors_chroma = np.sqrt(colors_hue_chroma[:,0]**2 + colors_hue_chroma[:,1]**2)
	colors_hue_chroma[colors_hue_chroma == 0] = 0.000001
	colors_hue = np.array(np.arctan(colors_hue_chroma[:,1]/colors_hue_chroma[:,0]))
	
	#computing mean value of dot between each person (shuffled) and the frequency with the bigest value for chroma
	#ind_corr_chroma.append(independent_correlation(colors_chroma,frequency_picture_books_reescaled))

	#compute the correlation matrix of chroma

	corr_mat = Corr_Mat(colors_chroma)

	#computing the eigenvalues of chroma
	corr_mat = np.reshape(corr_mat,(26,26))
	eigen = np.linalg.eig(corr_mat)
	eigenvalues = eigen[0]
	eigenvectors = eigen[1]
	corr_mat_eive_reescaled = eigenvectors/np.linalg.norm(eigenvectors,axis=0)

	#compute the dot product with picture books (bigest dot product in non shuffle data)
	Dot_chroma.append(np.dot(frequency_picture_books_reescaled,corr_mat_eive_reescaled[np.argmax(eigenvalues)]))
	
	#compute the correlation matrix of hue

	corr_mat = Corr_Mat(colors_hue)

	#computing the eigenvalues
	corr_mat = np.reshape(corr_mat,(26,26))
	eigen = np.linalg.eig(corr_mat)
	eigenvalues = eigen[0]
	eigenvectors = eigen[1]
	corr_mat_eive_reescaled = eigenvectors/np.linalg.norm(eigenvectors,axis=0)

	#compute the dot product with picture books (bigest dot product in non shuffle data)
	Dot_hue.append(np.dot(frequency_picture_books_reescaled,corr_mat_eive_reescaled[np.argmax(eigenvalues)]))
	
	#compute the correlation matrix of luminosity

	corr_mat = Corr_Mat(colors_luminosity)

	#computing the eigenvalues
	corr_mat = np.reshape(corr_mat,(26,26))
	eigen = np.linalg.eig(corr_mat)
	eigenvalues = eigen[0]
	eigenvectors = eigen[1]
	corr_mat_eive_reescaled = eigenvectors/np.linalg.norm(eigenvectors,axis=0)

	#compute the dot product with picture books (bigest dot product in non shuffle data)
	Dot_luminosity.append(np.dot(frequency_picture_books_reescaled,corr_mat_eive_reescaled[np.argmax(eigenvalues)]))
	
	
	
#ploting the histogram of the dot product eigenvalue of chroma corr mat	
weights = np.ones_like(np.array(Dot_chroma))/float(len(Dot_chroma))

test = stats.kstest((np.array(Dot_chroma) - np.mean(np.array(Dot_chroma)) )/np.std(np.array(Dot_chroma))  ,'norm')
pV = test[1]
plt.hist( np.array(Dot_chroma)   ,bins=20,label='mean = {}\n std = {}\np_value = {}'.format(np.mean(np.array(Dot_chroma)),np.std(np.array(Dot_chroma)),pV),weights=weights)


plt.plot(np.linspace(-1,1,100),0.05*stats.norm.pdf(np.linspace(-1,1,100),scale=np.std(np.array(Dot_chroma))))
plt.title('dot product between the maximun eigenvalue \n of the shuffle data and the frequency of letters in picture books')
plt.xlabel('value of the dot product')
plt.ylabel('frequency')
plt.legend()
plt.savefig('chroma/dotPicturesMaxEigShoffleHist.pdf')
plt.close()

#ploting the histogram of the dot product eigenvalue of hue corr mat	


weights = np.ones_like(np.array(Dot_hue))/float(len(Dot_hue))

test = stats.kstest((np.array(Dot_hue) - np.mean(np.array(Dot_hue)) )/np.std(np.array(Dot_hue))  ,'norm')
pV = test[1]
plt.hist( np.array(Dot_hue)   ,bins=20,label='mean = {}\n std = {}\np_value = {}'.format(np.mean(np.array(Dot_hue)),np.std(np.array(Dot_hue)),pV),weights=weights)

#plt.plot(np.linspace(-1,1,100),0.05*stats.norm.pdf(np.linspace(-1,1,100),scale=np.std(np.array(Dot_hue))))
plt.xlabel('dot product between the maximun eigenvalue \n of the shuffle data and the frequency of letters in picture books')
plt.ylabel('frequency')
plt.legend()
plt.savefig('hue/dotPicturesMaxEigShoffleHist.pdf')
plt.close()

#ploting the histogram of the dot product eigenvalue of luminosity corr mat	


weights = np.ones_like(np.array(Dot_luminosity))/float(len(Dot_luminosity))

test = stats.kstest((np.array(Dot_luminosity) - np.mean(np.array(Dot_luminosity)) )/np.std(np.array(Dot_luminosity))  ,'norm')
pV = test[1]
plt.hist( np.array(Dot_luminosity)   ,bins=20,label='mean = {}\n std = {}\np_value = {}'.format(np.mean(np.array(Dot_luminosity)),np.std(np.array(Dot_luminosity)),pV),weights=weights)

#plt.plot(np.linspace(-1,1,100),0.05*stats.norm.pdf(np.linspace(-1,1,100),scale=np.std(np.array(Dot_hue))))
plt.xlabel('dot product between the maximun eigenvalue \n of the shuffle data and the frequency of letters in picture books')
plt.ylabel('frequency')
plt.legend()
plt.savefig('luminosity/dotPicturesMaxEigShoffleHist.pdf')
plt.close()

############################################################################################################
#only chroma have a significant corr_eigenvector aligned with a frequency.  
############################################################################################################
