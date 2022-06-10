#carlos.soto362@gmail.com

##############################################################################
#See https://github.com/carlossoto362/QLS2021-2022Diploma/blob/main/RL_proyect/Proyect.pdf 
#for a description of the proyect
##############################################################################

import matplotlib.pyplot as plt
import numpy as np


def plot_state(state,i_endowments,n1,n2):
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
	
	plt.show()
	plt.close()
plot_state([(2/5,3/4),(3/5,1/4),1],[(1,0),(0,1)],5,4)

		
