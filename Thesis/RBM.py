#carlos.soto362@gmail.com

#First, download the data from emnist 
#add reference Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import time



def sigma(x):
	return 1/(1+np.exp(-x))

def p_h_j_given_v(b_j,w_j,v):
	'''
	computes the probability of one hiden layer given a vector v, "a" is the vector of the same dimention as v, b_j is a number, w_j is the column j of the matrix w. 
	'''
	return sigma(b_j + np.dot(v,w_j))

def reconstruction_h_given_v(a,b,w,v,end='False'):
	'''
	reconstruct the hiden layer given a vector v. 
	'''
	
	return np.array([    np.random.choice( [0,1], p=[ p_h_j_given_v(b[j],w[:,j],v),1-p_h_j_given_v(b[j],w[:,j],v) ] ) for j in range(len(b))])
	

def p_v_given_h(a_j,wT_j,h):
	'''
	computes the probability of one visible layer given a vector h, "b" is the vector of the same dimention as h, a_j is a number, wT_j is the column j of the matrix w.T. 
	'''
	Energys = np.array([ a_j*k + np.dot(h,wT_j)*k for k in range(256) ])
	
	p_v =  np.array([ np.exp(a_j*k + np.dot(h,wT_j)*k - np.mean(Energys)) for k in range(256) ])
	return p_v/np.sum(p_v)

def p_v_given_h_bias(a_j,wT_j,h):
	'''
	computes the probability of one visible layer given a vector h, "b" is the vector of the same dimention as h, a_j is a number, wT_j is the column j of the matrix w.T. 
	The bias comes from the aproximation of the values of v as real numbers from 0 to 1 as the value return from the sigma function. (advised given by https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf for binary variables. If using p instead dont afect the result, maybe it dosent afect that much the result using it in this case.)
	'''
	return sigma(a_j + np.dot(h,wT_j))

def reconstruction_v_given_h(a,b,w,h,end='False'):
	'''
	reconstruct the visible layer given a vector h. 
	'''
	return np.array([    np.random.choice( np.arange(256), p=p_v_given_h(a[j],w.T[:,j],h) ) for j in range(len(a))])
	
def reconstruction_v_given_h_bias(a,b,w,h,end='False'):
	'''
	reconstruct the visible layer given a vector h, but using p_v_given_h_bias. This will be bias, but will be faster.
	'''
	return np.array([    p_v_given_h_bias(a[j],w.T[:,j],h) for j in range(len(a))])

def CD_reconstruction(n,a,b,w,v):
	"""
	CD_n reconstruction, n is the number of iterations before the reconstruction is return, a, b and w are the paramiters, v is a visible layer data. 
	"""
	new_v = v
	for i in range(n):
		new_h = reconstruction_h_given_v(a,b,w,new_v)
		new_v = reconstruction_v_given_h(a,b,w,new_h)
	return new_v,new_h
	
def CD_reconstruction_bias(n,a,b,w,v): ###################!!!!!!!!!!!!!!!!!!!! this one is used to reduce time of computation. Is no bias for the case of binary data. 
	"""
	CD_n reconstruction, n is the number of iterations before the reconstruction is return, a, b and w are the paramiters, v is a visible layer data. 
	using reconstruction_v_given_h_bias. This will be bias, but will be faster.
	"""
	new_v = v
	for i in range(n):
		new_h = reconstruction_h_given_v(a,b,w,new_v)
		new_v = reconstruction_v_given_h_bias(a,b,w,new_h)
	return new_v,new_h	
	
	
def error_CD_n(n,a,b,w,v,samples):
	"""
	RSE of the reconstruction of a set of observations v. v is a list of vectors. (use a random sample of samples data)
	"""
	v_shuffle = list(v)
	random.shuffle(v_shuffle)
	v_shuffle = v_shuffle[:samples]
	RSE = 0
	#print('computing RSE')
	for i in (range(len(v_shuffle))):
		new_v,new_h = CD_reconstruction_bias(n,a,b,w,v_shuffle[i])
		RSE += np.dot(v_shuffle[i]-new_v,v_shuffle[i]-new_v)/(28*28)
	RSE = RSE/len(v_shuffle)
	return RSE
		

def proportion_on(v):
	return np.sum(v,axis=0)/len(v)



#training the RLM
class RBM():
	def __init__(self,learning_rate_weight,learning_rate_b,learning_rate_a, hiden_layers_size,minibatch_size,proportionOn,cd_n):
		
		self.alpha_w = learning_rate_weight
		self.alpha_a = learning_rate_a
		self.alpha_b = learning_rate_b
		self.m = minibatch_size
		self.l_h = hiden_layers_size
		self.l_v = 28*28
		
		#advised in https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf
		in_log = np.divide(proportionOn, (1-proportionOn), out=np.ones(proportionOn.shape, dtype=float), where=(1-proportionOn)!=0) #no division by zero p/(1-p)
		in_log[in_log==0]=1 # for not having log(0)
		self.a = np.log(in_log) 
		
		self.b = np.zeros(self.l_h)
		self.w = np.zeros((self.l_v,self.l_h))
		self.cd_n = cd_n
		
	def one_epoch(self,v):
		'''
		v is a set of vectors. This function will also return the RSE for training porposes, but will be the E_in. The E_out will be monitored each epoch. 
		'''
		print('computing one epoch...')
		n = len(v)
		#shuffle
		shuffle_v = list(v)
		random.shuffle(shuffle_v)
		shuffle_v = np.array(shuffle_v)
		m_number = int(n/self.m)
		m_remainding = int(n%self.m)
		
		#for m_number of minibatches
		
		E_in = []
		
		for i in tqdm(range(m_number)):
			mean_vhT_given_v = np.zeros((self.l_v,self.l_h))
			mean_vhT = np.zeros((self.l_v,self.l_h))
			mean_h_given_v = np.zeros(self.l_h)
			mean_h = np.zeros(self.l_h)
			mean_v = np.zeros(self.l_v)
			mean_v_given_v = np.zeros(self.l_v)
			
			
			
			for j in (range(i*self.m,(i+1)*self.m)):
				new_h_given_v = reconstruction_h_given_v(self.a,self.b,self.w,shuffle_v[j])
				mean_vhT_given_v +=  np.outer(shuffle_v[j], new_h_given_v )
				new_v,new_h = CD_reconstruction_bias(self.cd_n,self.a,self.b,self.w,shuffle_v[j])
				mean_vhT += np.outer(new_v,new_h)
				mean_h_given_v += new_h_given_v
				mean_h += new_h
				mean_v_given_v += shuffle_v[j]
				mean_v += new_v
				
			mean_vhT_given_v /= self.m
			mean_vhT /= self.m
			mean_h_given_v /= self.m
			mean_h /= self.m
			mean_v_given_v /= self.m
			mean_v /= self.m
			
			#divides by the size of the minibatch is an advise of https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf
			self.w += self.alpha_w*(mean_vhT_given_v - mean_vhT)/self.m
			self.a += self.alpha_a*(mean_v_given_v - mean_v)/self.m
			self.b += self.alpha_b*(mean_h_given_v - mean_h)/self.m
			
			
			
		#last minibatch
		if m_remainding != 0:
			mean_vhT_given_v = np.zeros((self.l_v,self.l_h))
			mean_vhT = np.zeros((self.l_v,self.l_h))
			mean_h_given_v = np.zeros(self.l_h)
			mean_h = np.zeros(self.l_h)
			mean_v = np.zeros(self.l_v)
			mean_v_given_v = np.zeros(self.l_v)
			for j in range(m_number*self.m,m_number*self.m + m_remainding):
				new_h_given_v = reconstruction_h_given_v(self.a,self.b,self.w,shuffle_v[j])
				mean_vhT_given_v +=  np.outer(shuffle_v[j], new_h_given_v )
				new_v,new_h = CD_reconstruction_bias(self.cd_n,self.a,self.b,self.w,shuffle_v[j])
				mean_vhT += np.outer(new_v,new_h)
				mean_h_given_v += new_h_given_v
				mean_h += new_h
				mean_v_given_v += shuffle_v[j]
				mean_v += new_v
					
			mean_vhT_given_v /= self.m
			mean_vhT /= self.m
			mean_h_given_v /= self.m
			mean_h /= self.m
			mean_v_given_v /= self.m
			mean_v /= self.m
			
			self.w += self.alpha_w*(mean_vhT_given_v - mean_vhT)/m_remainding
			self.a += self.alpha_a*(mean_v_given_v - mean_v)/m_remainding
			self.b += self.alpha_b*(mean_h_given_v - mean_h)/m_remainding
		
		
		return error_CD_n(self.cd_n,self.a,self.b,self.w,shuffle_v,100)

