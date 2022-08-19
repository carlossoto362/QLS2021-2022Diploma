#carlos.soto362@gmail.com

#First, download the data from emnist 
#add reference Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import time
from RBM import *

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
imagesTest = imagesTest = np.reshape(imagesTest,(len(imagesTest),28*28))


##########################################################################################################################################################################
#lets use the same fraction of data as frequency_picture_books
#[16,23,25,9,21,10,5,1,15,2,22,12,24,6,20,11,3,17,18,8,7,13,0,14,19,4] = [q,x,z,j,v,k,f,b,p,c,w,m,y,g,u,l,d,r,s,i,h,n,a,o,t,e]
##########################################################################################################################################################################
frequency_picture_books = np.array([7.94,1.54,2.07,4.47,11.48,1.52,2.41,5.92,5.65,0.14,1.33,4.35,2.28,6.14,7.97,1.57,0.07,5.15,5.54,8.06,3.01,0.77,2.24,0.12,2.28,0.13])
def fraction_letters(l):
	return np.array( [ len(l[l==i]) for i in range(26) ] )
str_colors = ['a','b','c','d','e','f','g','h', 'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

#frequency_order = [16,23,25,9,21,10,5,1,15,2,22,12,24,6,20,11,3,17,18,8,7,13,0,14,19,4]	
#data_plot = list(zip(fraction_letters(labelsTrain[:50000]),str_colors ))
#data_plot.sort(key = lambda val: val[0])
#data_plot = np.array(data_plot).T

	

#make a list of elements that match the frequency_picture_books frequencys, the only exeptions are
# i, o and s, which don't have the necesary amount of letters (I whant to do around 50000 data points.)
frequency_order = np.arange(26)
frequency_size = (frequency_picture_books*50000/np.sum(frequency_picture_books)).astype(int)
#print(np.sum(frequency_size))

imagesTrain_use = np.zeros((49985-275-1483-243,28*28))
labelsTrain_use = np.zeros(49985-275-1483-243)

index = 0
for i in range(26):
	if frequency_size[i] <= len(labelsTrain[labelsTrain == frequency_order[i]]):
		imagesTrain_use[index:index + frequency_size[i]]  =  (imagesTrain[labelsTrain == frequency_order[i]])[:frequency_size[i]]   
		labelsTrain_use[index:index + frequency_size[i]]  =  (labelsTrain[labelsTrain == frequency_order[i]])[:frequency_size[i]]
		index = index + frequency_size[i]
	else:
		#print(frequency_size[i-1],frequency_size[i],len(labelsTrain[labelsTrain == frequency_order[i]]),str_colors[i])
		imagesTrain_use[index:index + len(labelsTrain[labelsTrain == frequency_order[i]])]  =  (imagesTrain[labelsTrain == frequency_order[i]])  
		labelsTrain_use[index:index + len(labelsTrain[labelsTrain == frequency_order[i]])]  =  (labelsTrain[labelsTrain == frequency_order[i]])
		index = index + len(labelsTrain[labelsTrain == frequency_order[i]])
	

imagesTrain = imagesTrain_use
labelsTrain = labelsTrain_use
#print('done 1')



#############################################################################################################################################################################
#############################################################################################################################################################################
#imagesTrain = imagesTrain[:50000]   ################!!!!!!!!!!!!!!!!!!! to reduce time of computation
#imagesTest = imagesTest[:5000]


#transforming data in binary, document 2 is without round

imagesTrain = (imagesTrain/np.max(imagesTrain)).round(0)
imagesTest = (imagesTest/np.max(imagesTest)).round(0)

###############################################################################################################################################################################
#Let's train the data. Using https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf, 
#am going to train the data with less training letters, for diferent values of minibatch size, and see which is the most efitient.  
###############################################################################################################################################################################

learning_rate_weight = np.random.normal(loc=0,scale=0.01)
learning_rate_b = np.random.normal(loc=0,scale=0.01)
learning_rate_a = np.random.normal(loc=0,scale=0.01)

#supossing each image is iqually likly, a good representation would use H bits to encode one image, H = 784 bits (log(2^(28*28))). The advised is then around 126, am going to use only 50 because is more or less of the same order. 
hiden_layers_size = 50

#https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf advised around 100
minibatch_size = 40






for miniBatchSize in [40]:
	Letters_RBM = RBM(learning_rate_weight,learning_rate_b,learning_rate_a,hiden_layers_size,miniBatchSize,proportion_on(imagesTrain),2)
	E_in = []
	E_out = []
	
	E_in.append(error_CD_n(2,Letters_RBM.a,Letters_RBM.b,Letters_RBM.w,imagesTrain,100))
	E_out.append(error_CD_n(2,Letters_RBM.a,Letters_RBM.b,Letters_RBM.w,imagesTest,100))
	
	for epoch in (range(100)): ##########################!!!!!!!!!!! only 10 to reduce time of computation
		E_in.append(Letters_RBM.one_epoch(imagesTrain))
		E_out.append(error_CD_n(2,Letters_RBM.a,Letters_RBM.b,Letters_RBM.w,imagesTest,100))
		Letters_RBM.alpha_w = np.random.normal(loc=0,scale=0.01)
		Letters_RBM.alpha_a = np.random.normal(loc=0,scale=0.01)
		Letters_RBM.alpha_b = np.random.normal(loc=0,scale=0.01)
		
	plt.plot(np.arange(101),E_in,'--',color='red',label='In Sampling RSE')
	plt.plot(np.arange(101),E_out,'--',color='blue',label='E_out RSE, minibatches size: {}'.format(miniBatchSize))
	plt.xlabel('epochs')
plt.title('Root Square Error')
plt.legend()
plt.savefig('RBM_RSE_picture_books.pdf')
plt.close()
	


#if is ok, then, let's store the final result. 
file = open('RBM_paramiters_picture_books.txt','w')
file.write('#{} hiden units'.format(hiden_layers_size) + ' first a, then b, last w.\n')
file.write(','.join(str(e) for e in Letters_RBM.a))
file.write(';')
file.write(','.join(str(e) for e in Letters_RBM.b))
file.write(';')
file.write(','.join(str(e) for e in np.reshape(Letters_RBM.w,(Letters_RBM.w.shape[0]*Letters_RBM.w.shape[1]))))
file.close()
		
		 
	

#plot
	
#imagesTrain = np.reshape(imagesTrain,(len(imagesTrain),28,28))
#imagesTest = imagesTest = np.reshape(imagesTest,(len(imagesTest),28,28))

#plt.imshow(imagesTrain[0],cmap='gray')
#plt.show()
