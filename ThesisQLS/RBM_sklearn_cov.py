#carlos.soto362@gmail.com

#First, download the data from emnist 
#add reference Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import time
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.neural_network import BernoulliRBM

from emnist import extract_training_samples
from emnist import extract_test_samples



#############################################################################################################################################################################
#letters colores with 28x28 pixels, each pixel has a value between 0 and 255. 163939 train, and 27059 test. labels are the letters (lowercase) [a,b,..,z] -> [0,1,...,25].
#am going to use only 50000 for training, and am going to transform the value of the pixels binary values 
#############################################################################################################################################################################
imagesTrain, labelsTrain = extract_training_samples('byclass')
imagesTest,labelsTest = extract_test_samples('byclass')
imagesTrain = imagesTrain[labelsTrain>=36]
labelsTrain = labelsTrain[labelsTrain>=36] - 36
imagesTest = imagesTest[labelsTest>=36]
labelsTest = labelsTest[labelsTest>=36] -36


#Transforming the data in vectors and biceversa. For time issues, there is not going to be used all the data. 

imagesTrain = np.reshape(imagesTrain,(len(imagesTrain),28*28))
imagesTest = np.reshape(imagesTest,(len(imagesTest),28*28))

#same amount of letters for testing (p(l) = 1/26). only use data not used for training.
frequency_picture_books = np.array([7.94,1.54,2.07,4.47,11.48,1.52,2.41,5.92,5.65,0.14,1.33,4.35,2.28,6.14,7.97,1.57,0.07,5.15,5.54,8.06,3.01,0.77,2.24,0.12,2.28,0.13])
frequency_order = np.arange(26)
frequency_size = (frequency_picture_books*50000/np.sum(frequency_picture_books)).astype(int)

#print(len(imagesTrain))

imagesTrain_use = np.zeros((163939 - 49985+275+1483+243,28*28))# i, o and s, which don't have the necesary amount of letters (I whant to do around 50000 data points.)
labelsTrain_use = np.zeros(163939 - 49985+275+1483+243)

index = 0
for i in range(26):
	le = len((imagesTrain[labelsTrain == frequency_order[i]])[frequency_size[i]:])
	imagesTrain_use[index:index + le]  =  (imagesTrain[labelsTrain == frequency_order[i]])[frequency_size[i]:]
	labelsTrain_use[index:index + le]  =  (labelsTrain[labelsTrain == frequency_order[i]])[frequency_size[i]:]
	index = index + le	
	
print('done part 1')

imagesTrain = imagesTrain_use
labelsTrain = labelsTrain_use
#getting more space of memory
imagesTrain_use = None
labelsTrain_use = None

images = np.zeros((len(imagesTrain) + len(imagesTest), 28*28))
labels = np.zeros((len(imagesTrain) + len(imagesTest),))

images[:len(imagesTrain)]=imagesTrain
images[len(imagesTrain):] = imagesTest
labels[:len(imagesTrain)]=labelsTrain
labels[len(imagesTrain):] = labelsTest

#getting more space of the memory
imagesTrain = None
labelsTrain = None


#use only 250 images per data tipe (13 hours aprox)
print('done part 1')
imagesTest_use = np.zeros((50*26,28*28))
labelsTest_use = np.zeros(50*26)
for i in range(26):
	if len(images[labels == i]) < 50:
		print('less',i)
	else:
		imagesTest_use[i*50:(i+1)*50]  =  images[labels == i][:50]   
		labelsTest_use[i*50:(i+1)*50]  =  labels[labels == i][:50]
	


imagesTest = imagesTest_use
labelsTest = labelsTest_use


#getting more space of the memory
imagesTest_use = None
labelsTest_use = None

#transforming data in binary
imagesTest = (imagesTest/np.max(imagesTest)).round(0)
print('done part 2')

#################################################################################################################################################################################
#download the parameters learned from the rbm
#################################################################################################################################################################################
file = open('RBM_paramiters_sklearn.txt','r')
file.readline()
a_b_w = file.readline()
a_b_w = a_b_w.rsplit(';')
a_b_w[0] = a_b_w[0].rsplit(',')
a_b_w[1] = a_b_w[1].rsplit(',')
a_b_w[2] = a_b_w[2].rsplit(',')
file.close()




a = np.array(a_b_w[0]).astype(float)
b = np.array(a_b_w[1]).astype(float)
w = np.array(a_b_w[2]).astype(float).reshape((28*28,30))

'''
for i in range(2):
	new_v,new_h = CD_reconstruction_bias(10,a,b,w,imagesTest[10*i])
#new_v[new_v<0.5] = 0
#new_v[new_v>=0.5] = 1

	plt.imshow(imagesTest[10*i].reshape((28,28)),cmap='gray')
	plt.show()
	plt.close()
	plt.imshow(new_v.reshape((28,28)),cmap='gray')
	plt.show()
	plt.close()
'''
#################################################################################################################################################################################
#find the best internal representation given a letter-label. 
#################################################################################################################################################################################
def s_max(b,w,v):
	s = np.zeros(len(b))
	s[(b+np.dot(v,w))>0] = 1
	return s
	

#################################################################################################################################################################################
#computing the join probability estimate
#################################################################################################################################################################################

def p_s(s):
	'''
	estimate of p(s=s') as \sum_{N images} \delta_{s=s'}
	'''
	ps = 0
	for image in imagesTest:
		if (s == s_max(b,w,image)).all():
			ps += 1
	ps /= len(imagesTest)
	return ps

def p_l_given_s(s,l):
	'''
	estimate of p(s=s') as \sum_{N images} \delta_{s=s'}\delta_{l=l'}
	'''
	psl = 0
	for i in range(len(imagesTest)):
		if (s == s_max(b,w,imagesTest[i])).all() and (l == labelsTest[i] ):
			psl += 1
	psl /= len(imagesTest)
	return psl
	
p_ll = np.zeros((26,26))
images = imagesTest
labels = labelsTest
for i in tqdm(range(len(images))):
	s = s_max(b,w,images[i])
	ps = p_s(s)
	psl = np.zeros(26)
	for l in range(26):
		psl[l] = p_l_given_s(s,l)
	p_ll += np.outer(psl,psl)/ps
p_ll /= (26**2*(len(images))**2)


#storing the joint probabilitys
Ofile = open('p_ll_sklearn'+'.txt','w')
for col in p_ll:
	for element in col:
		Ofile.write(str(element)+' ')
	Ofile.write('/n')
Ofile.close()

cmap = cm.get_cmap('plasma', 256)
psm = plt.pcolormesh(p_ll, cmap=cmap, rasterized=True, vmin=np.min(p_ll), vmax=np.max(p_ll))
plt.colorbar(psm)
plt.savefig('p_l_l_sklearn' + '.pdf')
plt.close()


#################################################################################################################################################################################
#looking for correlation with corr_map_Lab
#################################################################################################################################################################################


def corr_mat_vrs_pll(corr_mat_,p_ll_,name) :
	

	corr_mat_ = corr_mat_.reshape((26,26))
	il2 = np.tril_indices(26, -1)
	corr_mat_ = corr_mat_[il2]
	p_ll_ = p_ll_[il2]

	#simple linear regresion and kendall tau
	x_plot = np.linspace(np.min(corr_mat_),np.max(corr_mat_),100)
	linearRegression = scipy.stats.linregress(corr_mat_,p_ll_ )
	k_t = scipy.stats.kendalltau(corr_mat_,p_ll_ )
	plt.plot(corr_mat_,p_ll_,'o',color='green')
	plt.plot(x_plot,linearRegression.slope * x_plot + linearRegression.intercept, color='blue',label = '{:.2E}x+{:.2E}\nr^2={:.2f}\nk_t.statistic={:.2f}\nk_t.p_value={:.2E}'.format(linearRegression.slope,linearRegression.intercept,linearRegression.rvalue,k_t[0],k_t[1]))
	plt.xlabel('Lower triangel of the Correlation Matrix')
	plt.ylabel('Lower Triangel of Reescaled joint pdf')
	plt.legend()

	plt.savefig('corr_mat_vrs_pll/'+ name + '.pdf')
	plt.close()
	
f = open('corr_mat/'+ 'corr_mat' + '.txt','r')
corr_mat = np.array(f.read().replace('/n','').split()).astype('float')
f.close()

corr_mat_vrs_pll(corr_mat,p_ll,'corr_matLabVsP_ll_sklearn.pdf')


f = open('corr_mat/covMatrixLabCie2000.txt','r')
corr_mat = np.array(f.read().replace('/n','').split()).astype('float')
f.close()

corr_mat_vrs_pll(corr_mat,p_ll,'corr_mat_Labcie2000VsP_ll_sklearn.pdf')

f = open('luminosity/corr_mat.txt','r')
corr_mat = np.array(f.read().replace('/n','').split()).astype('float')
f.close()

corr_mat_vrs_pll(corr_mat,p_ll,'corr_mat_luminosityVsP_ll_sklearn.pdf')

f = open('hue/corr_mat.txt','r')
corr_mat = np.array(f.read().replace('/n','').split()).astype('float')
f.close()

corr_mat_vrs_pll(corr_mat,p_ll,'corr_mat_hueVsP_ll_sklearn.pdf')

f = open('chroma/corr_mat.txt','r')
corr_mat = np.array(f.read().replace('/n','').split()).astype('float')
f.close()

corr_mat_vrs_pll(corr_mat,p_ll,'corr_mat_chromaVsP_ll_sklearn.pdf')
	
