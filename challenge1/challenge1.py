#author: Carlos Enmanuel Soto Lopez (carlos.soto362@gmail.com)
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
#This program is designed to aniliced two sets of data (A.fna and B.fna), one is a DNA chain of ecola, the second one is
#a random realisation of the same components of the DNA chain. The idea is to identify the correct chain.
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################



import numpy as np
import matplotlib.pyplot as plt

def imList(name):
	file = open(name,'r') 

	data = []
	for line in file.readlines():
		l = list(line)
		l.pop(-1)
		data = data+l
	file.close()
	return data
	
def frequencies(data_,k):
	"""
	data_ is a set of letters. k is the lenght of a word, frequencies(data_,k) returns a Dictionary with words as keys
	and the number of ocurrences as values. 
	"""
	N = len(data_)
	frequencies_ = {}
	flag = 0
	word = ''
	for i in range(N):
		word += data_[i]
		flag += 1
		if flag == k:
			frequencies_[word] = frequencies_.get(word,0) +  1
			word = ''
			flag = 0
	return frequencies_

def frequenciesM(frequencies_):
	"""
	frequencies_ is a Dictionary with words as keys and frequencies as values. frequenciesM counts the number 
	of words with the same frequency and returns a dictionary with frequencie as keys and number of ocurrences as values.
	"""

	frequenciesM_ = {}
	for data_ in frequencies_:
		frequenciesM_[frequencies_[data_]] = frequenciesM_.get(frequencies_[data_],0) + 1
	return frequenciesM_
	
def entropy(frequencies_):
	"""
	frequencies is a Dictionary with elements as keys and frequencies*N as values. entropy(frequencies_) returns H= /sum_elements f_i log(1/f_i)
	"""
	N=np.sum(   np.array(list(zip(*frequencies_.items()))[1])   )
	#print('N', N)
	
	H=0
	for element in frequencies_:
		H -= (frequencies_[element]/N) * np.log((frequencies_[element]/N))
	
	return H
##########################################################################################################################
#Let's try to analyze the data without previous knowledge. So, both are a list of letters with four characters,
#A,C,G,T. What is the frequencies of this letters in both of the data sets?	
	

dataA = imList('A.fna')
dataB = imList('B.fna')
N=len(dataA)
frequenciesA = {'A' : dataA.count('A'), 'C':dataA.count('C'),'T':dataA.count('T'),'G':dataA.count('G')} 
frequenciesB = {'A' : dataB.count('A'), 'C':dataB.count('C'),'T':dataB.count('T'),'G':dataB.count('G')} 

sep='''
##############################################
##############################################
##############################################
'''
print(sep)
print('number of ocurrences for each letter, set A: ',frequenciesA,'\n number of ocurrences for each letter, set B: ',frequenciesB)

##########################################################################################################################
#Looks like both of this sets of data have the same amount of frequencies. The idea is to recognise which of this sets are
#a DNA list. So let's try to analyze the information contained on the list. I will try in two ways, first just
#computing the entropy of words of length k, assuming that the wrong set contains less information than the DNA set (is more random),
#the set with smallest entropy should be the correct set. 
#The second option is to compute the degenerate entropy or relevance, which quantifies the amount of information that is not noisy.
#In this sence, the entropy is the coding cost, and the relevance is the information codified. So, the right set is the one that
#maximised the relevance H[k] given the coding cost H[s].

k=10
frequenciesWA=frequencies(dataA,k)
frequenciesWB=frequencies(dataB,k)

HA = entropy(frequenciesWA)
HB = entropy(frequenciesWB)

print(sep)
print('entropy using {} letters per word, set A: '.format(k),HA,'\n entropy using {} letters per word, set B: '.format(k),HB)

if HA < HB:
	r = 'A'
else:
	r = 'B'

print('the correct set seems to be the set {}'.format(r) )


frequenciesMA = frequenciesM(frequenciesWA)
frequenciesMB = frequenciesM(frequenciesWB)




HkA = entropy(frequenciesMA)
HkB = entropy(frequenciesMB)


print(sep)
print('fraction of information excluding noise (H[k]/H[s]), using {} letters per word, set A: '.format(k),HkA/HA,'\n fraction of information excluding noise using {} letters per word, set B: '.format(k),HkB/HB)

if (HkA/HA) > (HkB/HB):
	r = 'A'
else:
	r = 'B'

print('using the total information, the correct set seems to be the set {}'.format(r) )


print(sep)
#######################################################################################################
#######################################################################################################
#######################################################################################################
#now let's make a graph of the current knowledge we have. using k small, e.g. k= 4, it's easier to understand the meaning of the axis.  
#the last graph shows that the set A has a more wider spread of frequencies, while the set B is more concentrated.
#Using k big, e.g. k =10, and log scale, its posible to distinwish that the set A is closer to the zips law. 

frequenciesWAp = [[],[]]
i=1
for element in frequenciesWA:
	frequenciesWAp[0].append(i)
	frequenciesWAp[1].append(frequenciesWA[element])
	i+=1

frequenciesWBp = [[],[]]
i=1
for element in frequenciesWB:
	frequenciesWBp[0].append(i)
	frequenciesWBp[1].append(frequenciesWB[element])
	i+=1

plt.ylabel('number of ocurrences')
plt.xlabel('words index')

plt.bar(frequenciesWAp[0],frequenciesWAp[1],label='set A')


plt.bar(frequenciesWBp[0],frequenciesWBp[1],alpha=0.4,label='set B')

plt.legend()
plt.savefig('frequenciesk10.pdf')
plt.clf()




frequenciesMAp = np.array(list(zip(*frequenciesMA.items())))
frequenciesMBp = np.array(list(zip(*frequenciesMB.items())))

plt.yscale('log')
plt.ylabel('Number of words m_k')
plt.xlabel('Number of ocurrences of a word [N*f]')

plt.bar(frequenciesMAp[0],frequenciesMAp[1],label='set A')


plt.bar(frequenciesMBp[0],frequenciesMBp[1],alpha=0.4,label='set B')

plt.plot(frequenciesMAp[0],frequenciesMAp[1,0]*np.array([(1/x)**2 for x in frequenciesMAp[0]]),label='{}x^(-2)'.format(4**10))

plt.legend()
plt.savefig('frequenciesMk10.pdf')



	


