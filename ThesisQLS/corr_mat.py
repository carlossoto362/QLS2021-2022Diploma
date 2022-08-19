#carlos.soto362@gmail.com

#this program is to create the correlation matrix $Corr(x,x') = (1/N)\sum_{n} (||C_x,C_x'||/||C_x||*||C_x'||), where $||a,b||$ refers to
#the norm, and $C_x$ is the color a synesthete perceives for the graphem $x$. The sum is over all the synesthetes sample. 
#The plan is to try diferent norms.  
#u_rgb is on sRGB color maping, u_rlab is in CIE L*a*b* mapping. 

#CREATES 4 SETS OF DATA AND THEIR RESPECTIVE GRAPHYC REPRESENTATION OF DIFERENT WAYS OF DEFINDING THE NORMALICED COVARIANCE MATRIX.


from pymatreader import read_mat
import numpy as np
from colormath.color_objects import LabColor,sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import math 
from matplotlib import cm
import matplotlib.pyplot as plt


#srgb = sRGBColor(*np.array([0.99810533, 0.0065083, 0.00615477]))   	#function to convert from rgb type array to Standart RGB mapping. 
#lab = convert_color(srgb, LabColor)					#function to convert from srgb to labcolor
#rgb = srgb.get_upscaled_value_tuple()					#function to get from srgb the 255 eight bits maping


data = read_mat('Synesthesia-v1.0/eagleman25-Synesthesia-e06c20b/data/EaglemanColoredAlphabets.mat')
#colors = np.delete(np.array(data['u_rgb']),0,1)

#print(colors[i].T[j]) i diferent people, j diferent letter. 

print(data.keys())
def p_norm(x,p=2):
	"""
	x is an n dimentional vector, p_norm returns the p-norm of this vector
	"""
	return (np.sum(np.array(x)**p))**(1/p)
	
def dist(x,y,p=2,typ='p'):
	"""
	x, y vectors, dist returns the distance between x and y, defined as the (\sum_i (x_i - y_i)^p)^{1/p} if typ = "p", or the cie2000 distance if typ = "cie2000".
	"""
	if typ == 'p':
		return p_norm(np.abs(np.array(x)-np.array(y)),p)
	elif typ == 'cie2000':
		labX = LabColor(*x)
		labY = LabColor(*y)
		return delta_e_cie2000(labX,labY)

def angle(x,y):
	"""
	angle(x,y) returns x y^T/sqrt(x x^T y y^T).
	"""
	return np.dot(x,y) /np.sqrt(np.dot(x,x)*np.dot(y,y))
		

def norm_cov(x,y,p=2,typ='n'):
	"""
	norm_cov returns the covariance of x and y after converting x to mean cero and variance 1,  norm_cov = cov(x-mean(x),y - mean(y))/sqrt(var(x-mean(x))var(y-mean(y))), where cov(x,y) is calculated as (1/N)\sum_i 		x_i y_i^T, and var(x) = (1/N)\sum_i x_i x_i^T.
	When typ = "p" or "cie2000", (x_i-mean(x)) (y_j - mean(y))^T is replaced with dist(x_i,mean(x))*dist(y_j,mean(y)) cos(\ theta), where \ theta is the angle between (x_i-mean(x)) and (y_j - mean(y)).
	and the function dist(x,y,typ="p") or dist(x,y,p,typ="cie2000") is used togueder with angle(x,y).
	"""	
	x_ = np.array(x)
	y_ = np.array(y)
	
	x_ = np.array([np.array(['n','n','n']) if math.isnan(co[0]) else co for co in x_])
	bol = (x_ != np.array(['n','n','n']))
	x_ = x_[bol]
	x_ = np.reshape(x_,(int(len(x_)/3), 3)).astype('float')
	y_ = y_[bol]
	y_ = np.reshape(y_,(int(len(y_)/3), 3)).astype('float')
	
	y_ = np.array([np.array(['n','n','n']) if math.isnan(co[0]) else co for co in y_])
	bol = (y_ != np.array(['n','n','n']))

	if np.shape(bol) == ():
		pass
	else:
		y_ = y_[bol]
		y_ = np.reshape(y_,(int(len(y_)/3), 3)).astype('float')
		x_ = x_[bol]
		x_ = np.reshape(x_,(int(len(x_)/3), 3)).astype('float')
	
	n=len(x_)
	
	mx = np.zeros(np.shape(x_[0]))
	my = np.zeros(np.shape(y_[0]))
	
	for xi in x_:
		mx += np.array(xi)
	mx = mx/len(x_)
	
	for yi in y_:
		my += np.array(yi)
	my = my/len(y_)
	
	if typ == 'n':
		cov = np.sum(np.array([np.dot(x_[i]-mx,y_[i]-my) for i in range(n)]))
		varx = np.sum(np.array([np.dot(x_[i]-mx,x_[i]-mx) for i in range(n)]))
		vary = np.sum(np.array([np.dot(y_[i]-my,y_[i]-my) for i in range(n)]))
		
	else:
		cov = np.sum(np.array([dist( x_[i] , mx , p = p , typ = typ ) * dist (y_[i] , my , p = p , typ = typ ) * angle(x_[i] - mx,y_[i]-my) for i in range(n)]))
		varx = np.sum(np.array([ dist( x_[i] , mx , p = p , typ = typ )**2 for i in range(n)  ]))
		vary = np.sum(np.array([ dist( y_[i] , my , p = p , typ = typ )**2 for i in range(n)  ]))
		
		
	return cov/np.sqrt(varx*vary)
	
	
def imagCorr_Mat(name,array,p=1,cmap = 'plasma',typ='n'):
	

	colorsL = array
	corrMatrixL = np.zeros((26,26))
	for j in range(26):
		print(j)
		for k in range(26):
			print(k)
			corrMatrixL[j,k] = norm_cov(colorsL[:].T[j].T,colorsL[:].T[k].T,p=p,typ = typ)
	cmap = cm.get_cmap(cmap, 256)
	psm = plt.pcolormesh(corrMatrixL, cmap=cmap, rasterized=True, vmin=np.min(corrMatrixL), vmax=np.max(corrMatrixL))
	plt.colorbar(psm)
	plt.savefig(name + '.pdf')
	plt.close()
	#Ofile = open(name+'.txt','w')
	#for col in corrMatrixL:
	#	for element in col:
	#		Ofile.write(str(element)+' ')
	#	Ofile.write('/n')
	#Ofile.close()
	
	print('image done')



#print(colors[:].T[0].T)   (()\()\...\())


imagCorr_Mat('covMatrixLab',np.delete(np.array(data['u_rlab']),0,1))
imagCorr_Mat('covMatrixRGB',np.delete(np.array(data['u_rgb']),0,1))
imagCorr_Mat('covMatrixLabCie2000',np.delete(np.array(data['u_rlab']),0,1),typ = 'cie2000')
imagCorr_Mat('covMatrixRGBp1',np.delete(np.array(data['u_rgb']),0,1),typ = 'p',p=1)


