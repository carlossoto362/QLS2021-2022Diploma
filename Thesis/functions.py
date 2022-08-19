
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
from tqdm import tqdm

def WignerDistribution(y,mean,sigma):
	'''
	returns the Wigner probability distribution function evaluated at x for the maximun eigenvalue of a covariance matrix M = XX.T form by matrices X of dimention mxn, m<n.
	'''
	x = 2*(y-mean-1)/(sigma*np.sqrt(26))
	
	return ((2/np.pi)/(2*sigma*np.sqrt(26)))*(1-x**2)**(1/2)
	

def plotEigenvalues(corr_m,name,label,color1='green',color2='red',w=20,wignerD='False',cap='False'):

	'''
	corr_m is a matrix of dimention 26x26, with entries less or equal than one, and diagonal equal to one. plotEigenvalues plot the eigenvalues of such a matrix, as well as computing the eigenvalues of a matrix with the same caracteristics, full of random entries drawn from a normal distribution with mean equal to the mean of the original matrix, and same variance. When WignerD is 'True', plots the semicircle law formulated by Wigner. When cap is 'True', plots the theoretical maximun value for the eigenvalue when the dimention of the matrix tends to infinity and the mean value of the entries is set to zero. 
	'''


	#computing the eigenvalues
	corr_m_ = np.reshape(corr_m,(26,26))
	#np.fill_diagonal(corr_m_,0)
	eigen = np.linalg.eig(corr_m_)
	eigenvalues = eigen[0]
	eigenvectors = eigen[1]
	indexing = np.argsort(eigenvalues)
	eigenvalues.sort()
	
	
	#computing mean and variance of aij
	aij = corr_m[corr_m<0.9]
	aij_mean = np.mean(aij)
	aij_sigma = np.std(aij)
	#computing a random matrix to compare
	eigenvaluesDistribution = np.zeros(26*1000)
	for i in range(1000):
		experiment = np.random.normal(aij_mean,np.sqrt(aij_sigma),(26,26))
		experiment = experiment*experiment.T/2
		np.fill_diagonal(experiment,1)
	
		eigen = np.linalg.eig(experiment)
		eigenvalues2 = eigen[0]
		eigenvectors2 = eigen[1]
		eigenvalues2.sort()
		eigenvaluesDistribution[i*26:(i+1)*26] = eigenvalues2
	
	#ploting the eigenvalues and their distribution
	yrange = [np.max(np.array([eigenvalues.max(),eigenvalues2.max()])), np.min(np.array([eigenvalues.min(),eigenvalues2.min()]))]
	yinterval = (yrange[0] - yrange[1])/20
	
	plt.ylim((yrange[1]-yinterval,yrange[0]+yinterval))
	
	plt.xlabel('eigen values in ascendent order')
	plt.plot(np.arange(26),eigenvalues,'o',label=label,color=color1)
	plt.plot(np.arange(26),eigenvalues2,'o',label='eigen values of random matrix',color=color2)
	
	weights = np.ones_like(eigenvalues)/float(len(eigenvalues))*w
	plt.hist(eigenvalues,color=color1,alpha=0.4,orientation='horizontal',bins=20,weights=weights)
	weights = np.ones_like(eigenvaluesDistribution)/float(len(eigenvaluesDistribution))*w
	plt.hist(eigenvaluesDistribution,color=color2,alpha=0.4,orientation='horizontal',bins=15,weights=weights)
	if wignerD == 'True':
		plt.plot(WignerDistribution(np.linspace(-2*aij_sigma*np.sqrt(26),2*aij_sigma*np.sqrt(26),100),aij_mean,aij_sigma),\
		np.linspace(-2*aij_sigma*np.sqrt(26),2*aij_sigma*np.sqrt(26),100),label='semi-circle law')
	#plt.plot(np.linspace(0,25,10),np.array([(25)*aij_mean + 1 + aij_sigma**2/aij_mean]*10),'--',label='theoretical mean of lambda 1',color='black')
	if cap == 'True':
		plt.plot(np.linspace(0,25,10),np.array([2*aij_sigma*np.sqrt(26)+1]*10),'--',label='theoretical cap for lambda max',color='grey')
	test = stats.kstest(eigenvalues,eigenvalues2)
	pV = test[1]
	plt.text(20,yrange[1]-3*yinterval,'p_value = %.2E'%pV)
	
	plt.grid('True')
	
	plt.legend()
	plt.savefig(name)
	plt.close()
	return eigenvalues, eigenvectors, indexing


def Corr_Mat(colors_):
	'''
	Computes the correlation matrix of dimention 26x26 for the set of data composed of a matrix of dimention Nx3x26. 
	'''
	corrMatrix = np.zeros((26,26))
	for j in range(26):
		for k in range(26):
			corrMatrix[j,k] = corr(colors_[j].T,colors_[k].T)
	return np.reshape(corrMatrix,(26*26,))

#by curiosity, let's compute the eigenvalues of the corr_matrix of dimention NxN:
def Corr_Mat_People(colors_):
	'''
	Computes the correlation matrix of dimention NxN for the set of data composed of a matrix of dimention Nx3x26. 
	'''
	N = len(colors_)
	#print(N)
	corrMatrix = np.zeros((N,N))
	for j in range(N):
		for k in range(N):
			corrMatrix[j,k] = corr(colors_[j].T,colors_[k].T)
	return np.reshape(corrMatrix,(N*N,))
	
def plot_histogram(data,bins,color,label):
	'''
	plots the histogram for the frequencies of values on the correlation matrix. It is still needed to save the image. 
	'''
	
	weights = np.ones_like(data)/float(len(data))
	n, bins, patches = plt.hist(data, bins=100, density=False,weights=weights, facecolor=color, alpha = 0.75, label=label)
	plt.xlabel('value of the correlation')
	plt.ylabel('frecuencies')
	plt.title('Histogram of frequencies of the correlation matrix')
	plt.xlim(-0.25, 1)
	plt.grid(True)
	
	#print('done histogram')

def mean(x):
	
	if np.shape(x[0]) == ():
		m = np.zeros(1)[0]
	else:
		m = np.zeros(len(x[0]))
	for xi in x:
		m += xi
	return m/len(x)
def var(x):
	'''
	suposed mean zero
	'''
	n = len(x)
	return np.sum(np.array([np.dot(x[i],x[i]) for i in range(n)]))
	
def corr(x_,y_):
	n = len(x_)
	x = x_ - mean(x_)
	y = y_ - mean(y_)
	return np.sum(np.array([np.dot(x[i],y[i]) for i in range(n)]))/(np.sqrt(var(x))*np.sqrt(var(y)))
	
def shuffleL(colors_):
	colorsSh = []

	for i in range(26):
		listS = list(colors_.T[i].T)
		random.shuffle(listS)
		colorsSh.append(np.array(listS).T)
	return np.array(colorsSh).T
	
