#carlos.soto362@gmail.com

##############################################################################
#See https://github.com/carlossoto362/QLS2021-2022Diploma/blob/main/RL_proyect/Proyect.pdf (1)
#for a description of the proyect
##############################################################################

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import scipy


def plot_state(state,i_endowments,n1,n2,name,sA_max,sB_max,title='',destiny=''):
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
	ax.scatter(sA_max[0][0],sA_max[0][1],c='red',label = 'uA_max',marker = 'x')
	ax.scatter(sB_max[0][0],sB_max[0][1],c='black',label = 'uB_max',marker = 'x')
	ax.scatter(wa1,wa2,c = 'red',label = 'Initial Endawment',marker = 'o')
	
	
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
	
	plt.title(title)
	plt.savefig(destiny + name)
	plt.close()
	
def plot_policy(policy,i_endowments,n1,n2,name,sA_max,sB_max,title='',destiny=''):
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
	w1 = wa1 + wb1
	w2 = wa2 + wb2
	x,y = np.meshgrid(np.linspace(0,w1,n1+1),np.linspace(0,w2,n2+1))
	u = np.zeros((n1+1,n2+1))
	v = np.zeros((n1+1,n2+1))
	for i in range(n1):
		for j in range(n2):
			option = np.argmax(policy[i,j])
			
			if policy[i,j,0] == policy[i,j,1] and policy[i,j,0] == policy[i,j,2]:
				u[i,j] = 0
				v[i,j] = 0
			elif option == 0:
				prob = np.exp(policy[i,j,0])/np.sum(np.exp(policy[i,j]))
				u[i,j] = -prob*np.cos(np.pi/4)
				v[i,j] = prob*np.sin(np.pi/4)
			elif option == 1:
				prob = np.exp(policy[i,j,1])/np.sum(np.exp(policy[i,j]))
				u[i,j] = prob*np.cos(np.pi/4)
				v[i,j] = -prob*np.sin(np.pi/4)
			else:
				u[i,j] = 0
				v[i,j] = 0
			
	
	ticks1 = np.arange(0,w1 + w1/n1,w1/n1)
	ticks2 = np.arange(0,w2 + w2/n2,w2/n2)
	
	fig , ax = plt.subplots()
	ax.quiver(x,y,u,v,scale=12*n1/w1)
	ax.set_xlabel('Xa1')
	ax.set_ylabel('Xa2')
	ax.set_xticks(ticks1)
	ax.set_xlim(0,w1)
	ax.set_ylim(0,w2)
	ax.set_yticks(ticks2)
	ax.grid()
	ax.set_facecolor((201/255, 228/255, 245/255))
	ax.scatter(sA_max[0][0],sA_max[0][1],c='red',label = 'uA_max',marker = 'x')
	ax.scatter(sB_max[0][0],sB_max[0][1],c='black',label = 'uB_max',marker = 'x')
	
	
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
	
	
	plt.title(title)
	plt.savefig(destiny + name)
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
		self.gammat = 1
		
	def FunctionNewStates(self,s):
		'''
		Posible new states taking budget constrain in consideration. (only fisible alocations should be reashable). s is the index of (xa1,xa2,p2).
		'''
		newStatesIndex = []
		newStates = []
		if s[2] == len(self.P)-1:
			p_options = [s[2],0, s[2] - 1]
		elif s[2] == 0:
			p_options = [s[2],s[2] + 1, len(self.P)-1]
		else:
			p_options = [s[2],s[2] + 1, s[2]-1]
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
		uA = utilityA([s[0]*self.eta1,s[1]*self.eta2])
		uB = utilityB([self.x1 - s[0]*self.eta1,self.x2 - s[1]*self.eta2])
		uA_new = utilityA([new_s[0]*self.eta1,new_s[1]*self.eta2])
		uB_new = utilityB([self.x1 - new_s[0]*self.eta1,self.x2 - new_s[1]*self.eta2])
		deltaUA = uA_new - uA
		deltaUB = uB_new - uB
		
		#compute the temporal diference learning
		deltaA = deltaUA +  np.dot(new_s,self.WA)*self.gamma - np.dot(s,self.WA)
		deltaB = deltaUB +  np.dot(new_s,self.WB)*self.gamma - np.dot(s,self.WB)

		self.WA = self.WA + self.alphaW*deltaA*s
		self.WB = self.WB + self.alphaW*deltaB*s
		
		I = np.zeros(self.TA.shape)
		I[(int(s[0]),int(s[1]),int(s[2]),A)] = 1
		self.TA = self.TA + self.gammat*self.alphaT*deltaA*I/policyA[A]
		self.TB = self.TB + self.gammat*self.alphaT*deltaB*I/policyB[A]
		self.gammat = self.gamma*self.gammat
		return new_s
		
		
		
	

		
		
		
class actorCriticSimplifiedEdgworthBox():
	def __init__(self,discretization,initial_endowments,gamma,alphaT,alphaW,p_2):
		"""
		For initial price 2 = 1, use initial_price = n2-1. For easier interpretability, use initial_endowment = ((a,b),(c,d)) with a,b,c,d natural numbers
		and a+c = n1, b+d = n2.
		Actor critic class defines all the functions requared to excecute the actor critic algorithm descrived in (1) for two consumers, with the simplification of having always the same price.
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
		self.p2 = p_2
		self.WA = np.zeros(2)
		self.TA = np.zeros((self.n1+1,self.n2+1,3))
		self.WB = np.zeros(2)
		self.TB = np.zeros((self.n1+1,self.n2+1,3))
		self.a = np.arange(3)
		self.gammat = 1
		
	def FunctionNewStates(self,s):
		'''
		Posible new states taking budget constrain in consideration. (only fisible alocations should be reashable). s is the index of (xa1,xa2).
		'''
		newStatesIndex = []
		newStates = []

		if self.p2<=self.n2 - 1:
			newStates.append([s[0]*self.eta1+self.eta1,s[1]*self.eta2-self.P[int(self.p2)]*self.eta1])
			newStatesIndex.append([s[0]+1,s[1] - self.P[int(self.p2)]*(self.eta1/self.eta2)])
		else:
			newStates.append([s[0]*self.eta1+(self.p2-self.n2 + 2)*self.eta1,s[1]*self.eta2-(self.p2-self.n2 + 2)*self.P[int(self.p2)]*self.eta1])
			newStatesIndex.append([s[0]+(self.p2-self.n2 + 2),s[1] - (self.p2-self.n2 + 2)*self.P[int(self.p2)]*(self.eta1/self.eta2)])

		
		if self.p2<=self.n2 - 1:
			newStates.append([s[0]*self.eta1-self.eta1,s[1]*self.eta2+self.P[int(self.p2)]*self.eta1])
			newStatesIndex.append([s[0]-1,s[1] + self.P[int(self.p2)]*(self.eta1/self.eta2)])
		else: 
			newStates.append([s[0]*self.eta1-(self.p2-self.n2 + 2)*self.eta1,s[1]*self.eta2+(self.p2-self.n2 + 2)*self.P[int(self.p2)]*self.eta1])
			newStatesIndex.append([s[0]-(self.p2-self.n2 + 2),s[1] + (self.p2-self.n2 + 2)*self.P[int(self.p2)]*(self.eta1/self.eta2)])

		newStates.append([s[0]*self.eta1,s[1]*self.eta2])
		newStatesIndex.append([s[0],s[1]])
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
				policyA.append(np.exp(A[k] - np.mean(A))) #A = self.Ti[*s], np.max(A) is to aboid big numbers, but the normalization makes it to work fine
			else:
				policyA.append(0)
		
		if np.sum(policyA)!=0:
			return policyA/np.sum(policyA)
		else: 
			return policyA

	
	def FunctionPolicy(self,s):
		'''
		Policy of the two people combined 
		'''
		#person A would decide one of 6 options
		newStates , newStatesIndex = self.FunctionNewStates(s)
		
		policyA = self.FunctionPolicyA(newStates,self.TA[int(s[0]),int(s[1])])
		policyB = self.FunctionPolicyA(newStates,self.TB[int(s[0]),int(s[1])])
		policy = np.zeros(3)
		policy[:2] = policyA[:2]*policyB[:2]
		policy[2] = 1- np.sum(policy[:2])
		return policy , newStatesIndex,policyA,policyB
		
		
	
	def get_newState_acording_to_Policy(self, s):
		"""
		The actions are exgenge good 2 from customer 1 to customer 2 at current price p_i, or at price p_(i+1), or at price p_(i-1),
		exgenge good 2 from customer 2 to customer 1 at current price p_i, at price p_(i+1) or at price p_(i-1) or staying.         
		"""
		policy, newStatesIndex,policyA,policyB = self.FunctionPolicy(s)
		A = np.random.choice(3, p=policy) 
		
		return newStatesIndex[A],A,policyA,policyB
		
		
	def single_step_update(self, s):
		"""
		Uses a single step to update the values.
		"""
		#take actions and observe the new state and rewards
		new_s,A,policyA,policyB = self.get_newState_acording_to_Policy(s)
		uA = utilityA([s[0]*self.eta1,s[1]*self.eta2])
		uB = utilityB([self.x1 - s[0]*self.eta1,self.x2 - s[1]*self.eta2])
		uA_new = utilityA([new_s[0]*self.eta1,new_s[1]*self.eta2])
		uB_new = utilityB([self.x1 - new_s[0]*self.eta1,self.x2 - new_s[1]*self.eta2])
		deltaUA = uA_new - uA
		deltaUB = uB_new - uB
		
		#compute the temporal diference learning
		deltaA = deltaUA +  np.dot(new_s,self.WA)*self.gamma - np.dot(s,self.WA)
		deltaB = deltaUB +  np.dot(new_s,self.WB)*self.gamma - np.dot(s,self.WB)

		self.WA = self.WA + self.alphaW*deltaA*s
		self.WB = self.WB + self.alphaW*deltaB*s
		
		I = np.zeros(self.TA.shape)
		I[int(s[0]),int(s[1]),A] = 1
		self.TA = self.TA + self.gammat*self.alphaT*deltaA*I/policyA[A]
		
		self.TB = self.TB + self.gammat*self.alphaT*deltaB*I/policyB[A]
		self.gammat = self.gamma*self.gammat
		return new_s
	def loss_function(self):
		'''
		this function only works with simetric initial endowments, same utility functions and unique price, in order for the competitive equilibrium to be in the midle of 
		Edgworth Box and the dinamics only on the diagonal. 
		'''
		loss = 0
		for i in range(self.n2):
			if self.n2-i < self.n2/2:
				loss += 1-np.exp(self.TA[self.n2-i,i,0])/np.sum(np.exp(self.TA[self.n2-i,i]))
			elif self.n2-i > self.n2/2:
				loss += 1-np.exp(self.TA[self.n2-i,i,1])/np.sum(np.exp(self.TA[self.n2-i,i]))
			else:
				loss += 1-np.exp(self.TA[self.n2-i,i,2])/np.sum(np.exp(self.TA[self.n2-i,i]))
		return loss
			
		

		
		
#Finding the actual equilibrium poing
def max_points(s_0,p_2,total_endowments):
	'''
	finds the maximun for two functions, given their constrains.
	'''
	#constrain : s_0[0] + s_0[1]*p_2  = s[0] + s[1]*p_2
	
	re = minimize(lambda x: - np.log(s_0[0] + s_0[1]*p_2 - x*p_2 + 0.4) - np.log(x + 0.4) , s_0[1])	
	s2_min = re.x[0]	
	s1_min = s_0[0] + s_0[1]*p_2 - s2_min*p_2
	return np.array([[s1_min,s2_min],[total_endowments[0] - s1_min,total_endowments[1] - s2_min]])
		
#let's try to plot a point
#plot_state([(2/5,3/4),(3/5,1/4),1],[(1,0),(0,1)],5,4)

#let's try to run a cople of states:
discretization = np.array([10,10])
initial_endowment = np.array([10,10])
gamma = 1
alphaT = 0.001
alphaW = 0.001


"""
#competitive enviroment with no equilibrium:
actor_Critic = actorCriticSimplifiedEdgworthBox(discretization,initial_endowment,gamma,alphaT,alphaW,9)

for episode in range(100):
	s0 = np.array([np.random.choice(discretization[0]),np.random.choice(discretization[1])])
	#s0 = np.array([10,0])
	s=np.array(s0)
	for t in range(10000):
		s = actor_Critic.single_step_update(s)
	
	s_max_A = max_points(s0,actor_Critic.P[9],initial_endowment)
	s_max_B = max_points(initial_endowment - s0,actor_Critic.P[9],initial_endowment)
	#print(s_max_A,s_max_B)
	#print(utilityA((s[0],s[1])),utilityB((10 - s[0],10-s[1])))
	plot_state([(s[0],s[1]),(10 - s[0],10-s[1]),9],[(s0[0],s0[1]),(10-s0[0],10-s0[1])],10,10,'episode_{}.jpg'.format(episode),s_max_A,s_max_B,destiny = 'gifSimplified/')
"""
	
#now with equilibrium:
#Let's find the best values for the parameters. Starting with alpha_T:

#It was found that alpha_T = 0.0009 whas good. 
#It was also found that any alpha_w in the range [0.00001,0.001] is equally good with alpha_T = 0.0009

for alpha in [0.0009]:	

	actor_Critic2 = actorCriticSimplifiedEdgworthBox(discretization,initial_endowment,gamma,0.0009,alpha,9)
	loss = []
	for episode in range(100):
		
		s00 = np.random.choice(discretization[0])
		s0 = np.array([s00,discretization[1]-s00])
		s0_v = np.array([s0[0]*actor_Critic2.eta1,s0[1]*actor_Critic2.eta2])
		s=np.array(s0)
		s_max_Av = max_points(s0_v,actor_Critic2.P[9],initial_endowment)
		s_max_Bv = max_points(initial_endowment - s0_v,actor_Critic2.P[9],initial_endowment)
		
		for t in range(10000):
			s = actor_Critic2.single_step_update(s)
		loss.append(actor_Critic2.loss_function())
		#print(actor_Critic2.loss_function())
		
		s_v = np.array([s[0]*actor_Critic2.eta1,s[1]*actor_Critic2.eta2])
		
		
		if episode > 80:
			title = 'Final curve'
		else:
			title = ''
		
		#print(s_max_A,s_max_B)
		#print(utilityA((s[0],s[1])),utilityB((10 - s[0],10-s[1])))
		#plot_state([(s_v[0],s_v[1]),(initial_endowment[0] - s_v[0],initial_endowment[1]-s_v[1]),19],[(s0_v[0],s0_v[1]),(initial_endowment[0]-s0_v[0],initial_endowment[1]-s0_v[1])],discretization[0],discretization[1],'episode_{}.jpg'.format(episode),s_max_Av,s_max_Bv,title,destiny = 'gifSimplifiedE/')
		
		#print(utilityA((s_v[0],s_v[1])),utilityB((10 - s_v[0],10-s_v[1])),utilityA(s_max_Av[0]))
			
	#plot_policy(actor_Critic2.TA,[(s0_v[0],s0_v[1]),(initial_endowment[0]-s0_v[0],initial_endowment[1]-s0_v[1])],discretization[0],discretization[1],'Policy.jpg',s_max_Av,s_max_Bv,title,destiny = 'gifSimplifiedE/')		
	fit = scipy.optimize.curve_fit(lambda x,a,b,c: a*np.exp(b*x) + c,  np.arange(100),  loss,  p0=(6, np.log(1/6)/100,1))
	plt.plot(np.arange(100),loss,'--',label = 'alpha_w = %.2E'%alpha + '\nalpha_T = %.2E'%0.0009)
	plt.plot(np.arange(100),fit[0][0]*np.exp(fit[0][1]*np.arange(100)) + fit[0][2],'-',label = 'y = %.2f'%fit[0][0] + 'exp(%.2f x)'%fit[0][1] + ' + %.2f'%fit[0][2])
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_function_exp.pdf')

'''

#let's use the parameters learned to find the a policy and make a gif of this process. 

actor_Critic2 = actorCriticSimplifiedEdgworthBox(discretization,initial_endowment,gamma,0.0009,0.001,9)
for episode in range(150):
	
	s00 = np.random.choice(discretization[0])
	s0 = np.array([s00,discretization[1]-s00])
	s0_v = np.array([s0[0]*actor_Critic2.eta1,s0[1]*actor_Critic2.eta2])
	s=np.array(s0)
	s_max_Av = max_points(s0_v,actor_Critic2.P[9],initial_endowment)
	s_max_Bv = max_points(initial_endowment - s0_v,actor_Critic2.P[9],initial_endowment)
	
	for t in range(10000):
		s = actor_Critic2.single_step_update(s)
	#print(actor_Critic2.loss_function())
	
	s_v = np.array([s[0]*actor_Critic2.eta1,s[1]*actor_Critic2.eta2])
	
	
	if episode > 140:
		title = 'Final curve'
	else:
		title = ''
	
	#print(s_max_A,s_max_B)
	#print(utilityA((s[0],s[1])),utilityB((10 - s[0],10-s[1])))
	plot_state([(s_v[0],s_v[1]),(initial_endowment[0] - s_v[0],initial_endowment[1]-s_v[1]),19],[(s0_v[0],s0_v[1]),(initial_endowment[0]-s0_v[0],initial_endowment[1]-s0_v[1])],discretization[0],discretization[1],'episode_{}.jpg'.format(episode),s_max_Av,s_max_Bv,title,destiny = 'gifSimplifiedE/')
	
	#print(utilityA((s_v[0],s_v[1])),utilityB((10 - s_v[0],10-s_v[1])),utilityA(s_max_Av[0]))
		
	plot_policy(actor_Critic2.TA,[(s0_v[0],s0_v[1]),(initial_endowment[0]-s0_v[0],initial_endowment[1]-s0_v[1])],discretization[0],discretization[1],'episode_{}.jpg'.format(episode),s_max_Av,s_max_Bv,destiny = 'gifSimplifiedP/')		
	
'''	

		
