#carlos.soto362@gmail.com

##############################################################################
#See https://github.com/carlossoto362/QLS2021-2022Diploma/blob/main/RL_proyect/Proyect.pdf (1)
#for a description of the proyect
##############################################################################

import matplotlib.pyplot as plt
import numpy as np


def plot_state(state,i_endowments,n1,n2,name):
	"""
	states is a vector of two tuplas and a number [(xa1,xa2),(xb1,xb2),p2], 
	i_endowments is the a vector of two tuplas [(wa1,wa2),(wb1,wb2)],
	n1 is the discretization over good 1, 
	n2 is the discretization over good 2. 
	"""
	wa1 = i_endowments[0][0]
	wa2 = i_endowments[0][1]
	wb1 = i_endowments[1][0]
	wb2 = i_endowments[1][1]
	xa1 = state[0][0]
	xa2 = state[0][1]
	xb1 = state[1][0]
	xb2 = state[1][1]
	p2 = state[2]
	w1 = wa1 + wb1
	w2 = wa2 + wb2
	
	ticks1 = np.arange(0,w1 + w1/n1,w1/n1)
	ticks2 = np.arange(0,w2 + w2/n2,w2/n2)
	
	fig , ax = plt.subplots()
	ax.set_xlabel('Xa1')
	ax.set_ylabel('Xa2')
	ax.set_xticks(ticks1)
	ax.set_xlim(0,w1)
	ax.set_ylim(0,w2)
	ax.set_yticks(ticks2)
	ax.grid()
	ax.scatter(xa1,xa2,s=100,marker = '^',c='green')
	ax.set_facecolor((201/255, 228/255, 245/255))
	
	ax2 = ax.twinx()
	ax2.set_ylim(0,w2)
	ax2.set_ylabel('Xb2')
	ax2.set_yticks(ticks2,labels=np.around(np.flip(ticks2),decimals = 1))
	
	ax3 = ax.twiny()
	ax3.set_xticks(ticks1,labels = np.around(np.flip(ticks1),decimals = 1))
	ax3.set_xlabel('Xb1')
	ax3.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
                   
	ax.text(-w1/10,-w2/10,'OA', horizontalalignment='left',fontsize='large', fontweight='bold')
	ax.text(w1+w1/20,w2 + w2/20,'OB', horizontalalignment='left',fontsize='large', fontweight='bold')
	ax.text(w1,0,'P2 = {}'.format(p2),horizontalalignment='right',fontweight='bold')
	
	plt.savefig('gift/' + name)
	plt.close()


#let's use simetric dinamics...

def utilityA(x):
	'''
	utility function of consumer 1. 
	'''
	return np.log(x[0] + 0.4) + np.log(x[1] + 0.4)
	
def utilityB(x):
	'''
	utility function of consumer 2. 
	'''
	return np.log(x[0] + 0.4) + np.log(x[1] + 0.4)


class actorCriticEdgworthBox():
	def __init__(self,discretization,initial_endowments,gamma,alphaT,alphaW):
		"""
		For initial price 2 = 1, use initial_price = n2-1. For easier interpretability, use initial_endowment = ((a,b),(c,d)) with a,b,c,d natural numbers
		and a+c = n1, b+d = n2.
		Actor critic class defines all the functions requared to excecute the actor critic algorithm descrived in (1) for two consumers.
		"""
		self.n1 = discretization[0]
		self.n2 = discretization[1]
		self.x1 = initial_endowments[0]
		self.x2 = initial_endowments[1]
		self.gamma = gamma
		self.alphaT = alphaT
		self.alphaW = alphaW
		self.eta1 = self.x1/self.n1
		self.eta2 = self.x2/self.n2
		self.P = np.array([*np.flip(np.arange(1,self.n2+1)),*(1/np.arange(2,self.n1+1))])*(self.eta2/self.eta1)
		self.p1 = 1
		self.WA = np.zeros(3)
		self.TA = np.zeros((self.n1+1,self.n2+1,self.n1 + self.n2 - 1,7))
		self.WB = np.zeros(3)
		self.TB = np.zeros((self.n1+1,self.n2+1,self.n1 + self.n2 - 1,7))
		self.a = np.arange(7)
		
	def FunctionNewStates(self,s):
		'''
		Posible new states taking budget constrain in consideration. (only fisible alocations should be reashable). s is the index of (xa1,xa2,p2).
		'''
		newStatesIndex = []
		newStates = []
		p_options = [s[2],s[2] + 1, s[2] - 1]
		for p in p_options:
			if p<=self.n2 - 1:
				newStates.append([s[0]*self.eta1+self.eta1,s[1]*self.eta2-self.P[int(p)]*self.eta1,p])
				newStatesIndex.append([s[0]+1,s[1] - self.P[int(p)]*(self.eta1/self.eta2),p])
			elif p < self.n1 + self.n2 - 2:
				newStates.append([s[0]*self.eta1+(p-self.n2 + 2)*self.eta1,s[1]*self.eta2-(p-self.n2 + 2)*self.P[int(p)]*self.eta1,p])
				newStatesIndex.append([s[0]+(p-self.n2 + 2),s[1] - (p-self.n2 + 2)*self.P[int(p)]*(self.eta1/self.eta2),p])
			else:
				newStates.append([-1,-1,p])
				newStatesIndex.append([-1,-1,p])
		for p in p_options:
			if p<=self.n2 - 1:
				newStates.append([s[0]*self.eta1-self.eta1,s[1]*self.eta2+self.P[int(p)]*self.eta1,p])
				newStatesIndex.append([s[0]-1,s[1] + self.P[int(p)]*(self.eta1/self.eta2),p])
			elif p < self.n1 + self.n2 - 2:
				newStates.append([s[0]*self.eta1-(p-self.n2 + 2)*self.eta1,s[1]*self.eta2+(p-self.n2 + 2)*self.P[int(p)]*self.eta1,p])
				newStatesIndex.append([s[0]-(p-self.n2 + 2),s[1] + (p-self.n2 + 2)*self.P[int(p)]*(self.eta1/self.eta2),p])
			else:
				newStates.append([-1,-1,p])
				newStatesIndex.append([-1,-1,p])
		newStates.append([s[0]*self.eta1,s[1]*self.eta2,p])
		newStatesIndex.append([s[0],s[1],p])
		return np.array(newStates),np.array(newStatesIndex)
		
	def FunctionPolicyA(self,new_states,A):
		'''
		Policy that person A would have given state x = (xa1,xa2,p2) with indexes s, and new posible states new_states,
		 taking into acount the budget constrain. A = self.Ti[*s]. 
		
		'''
		policyA = []
		k=0
		for state in new_states:
			if state[0] >= 0 and state[1] >= 0 and state[0] <= self.x1 and state[1] <= self.x2:
				policyA.append(np.exp(A[k] - np.max(A))) #A = self.Ti[*s], np.max(A) is to aboid big numbers, but the normalization makes it to work fine
			else:
				policyA.append(0)
		
		if np.sum(policyA)!=0:
			return policyA/np.sum(policyA)
		else: 
			return policyA

	
	def FunctionPolicy(self,s):
		'''
		Policy of the two people combined given  
		'''
		#person A would decide one of 6 options
		newStates , newStatesIndex = self.FunctionNewStates(s)
		
		policyA = self.FunctionPolicyA(newStates,self.TA[int(s[0]),int(s[1]),int(s[2])])
		policyB = self.FunctionPolicyA(newStates,self.TB[int(s[0]),int(s[1]),int(s[2])])
		policy = np.zeros(7)
		policy[:6] = policyA[:6]*policyB[:6]
		policy[6] = 1- np.sum(policy[:6])
		return policy , newStatesIndex,policyA,policyB
		
		
	
	def get_newState_acording_to_Policy(self, s):
		"""
		The actions are exgenge good 2 from customer 1 to customer 2 at current price p_i, or at price p_(i+1), or at price p_(i-1),
		exgenge good 2 from customer 2 to customer 1 at current price p_i, at price p_(i+1) or at price p_(i-1) or staying.         
		"""
		policy, newStatesIndex,policyA,policyB = self.FunctionPolicy(s)
		A = np.random.choice(7, p=policy) 
		if A != 6:
			return newStatesIndex[A],A,policyA,policyB
		else:
			return s,A,policyA,policyB
		
	def single_step_update(self, s):
		"""
		Uses a single step to update the values.
		"""
		#take actions and observe the new state and rewards
		new_s,A,policyA,policyB = self.get_newState_acording_to_Policy(s)
		uA = utilityA([new_s[0]*self.eta1,new_s[1]*self.eta2])
		uB = utilityB([self.x1 - new_s[0]*self.eta1,self.x2 - new_s[1]*self.eta2])
		
		#compute the temporal diference learning
		deltaA = uA +  np.dot(new_s,self.WA)*self.gamma - np.dot(s,self.WA)
		deltaB = uB +  np.dot(new_s,self.WB)*self.gamma - np.dot(s,self.WB)

		self.WA = self.WA + self.alphaW*deltaA*s
		self.WB = self.WB + self.alphaW*deltaB*s
		
		I = np.zeros(self.TA.shape)
		I[(int(s[0]),int(s[1]),int(s[2]),A)] = 1
		self.TA = self.TA + self.alphaT*deltaA*I/policyA[A]
		self.TB = self.TB + self.alphaT*deltaB*I/policyB[A]
		return new_s
		

		
		
#let's try to plot a point
#plot_state([(2/5,3/4),(3/5,1/4),1],[(1,0),(0,1)],5,4)

#let's try to run a cople of states:
discretization = np.array([10,10])
initial_endowment = np.array([10,10])
gamma = 0.1
alphaT = 0.0001
alphaW = 0.001

actor_Critic = actorCriticEdgworthBox(discretization,initial_endowment,gamma,alphaT,alphaW)

for episode in range(100):
	s0 = np.array([np.random.choice(discretization[0]),np.random.choice(discretization[1]),np.random.choice(discretization[0] + discretization[1]-1)])
	s=np.array(s0)
	for t in range(90000):
		s = actor_Critic.single_step_update(s)
	plot_state([(s[0],s[1]),(10 - s[0],10-s[1]),s[2]],[(s0[0],s0[1]),(10-s0[0],10-s0[1])],10,10,'episode_{}.jpg'.format(episode))
		
		
		
		

		
