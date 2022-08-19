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
print(len(colors))

#################################################################################################
#define the luminosity, hue and chroma
#################################################################################################

colors_luminosity = colors[:,0]
colors_hue_chroma = colors[:,1:3]

colors_chroma = np.sqrt(colors_hue_chroma[:,0]**2 + colors_hue_chroma[:,1]**2)

colors_hue_chroma[colors_hue_chroma == 0] = 0.000001		#hue  for b = 0 is not defined, so I used a 'b' very small instead as aproximation.
colors_hue = np.array(np.arctan(colors_hue_chroma[:,1]/colors_hue_chroma[:,0]))

print(len(colors),len(colors_chroma))
print(hola)

#################################################################################################
#files to store relevant information
#################################################################################################
file_luminosity = open('luminosity/relevant_information_luminosity.txt','w')
file_luminosity.write('#carlos.soto362@gmail.com\n')
file_luminosity.close()
file_hue = open('hue/relevant_information_hue.txt','w')
file_hue.write('#carlos.soto362@gmail.com\n')
file_hue.close()
file_chroma = open('chroma/relevant_information_chroma.txt','w')
file_chroma.write('#carlos.soto362@gmail.com\n')
file_chroma.close()

#################################################################################################
#write the frequency lists
#################################################################################################

frequency_picture_books = np.array([7.94,1.54,2.07,4.47,11.48,1.52,2.41,5.92,5.65,0.14,1.33,4.35,2.28,6.14,7.97,1.57,0.07,5.15,5.54,8.06,3.01,0.77,2.24,0.12,2.28,0.13])
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

#######################################################################################################################
#plot the frequencys vrs the colors, and compute the R^2
######################################################################################################################


def plot_correlation(frequency_list,color_list,name,y_label,x_label,color1,color2):
	data_plot = list(zip(frequency_list,color_list,str_colors))

	data_plot.sort(key = lambda val: val[0])

	data_plot = [[x[0] for x in data_plot],[x[1] for x in data_plot],[x[2] for x in data_plot]]



	plt.plot(data_plot[0],np.mean(np.array(data_plot[1]),axis=1),color = color1,label='mean value')
	plt.plot(data_plot[0],np.mean(np.array(data_plot[1]),axis=1),'o',color = color1)
	plt.fill_between(data_plot[0], np.mean(np.array(data_plot[1]),axis=1) - np.std(np.array(data_plot[1]),axis=1), np.mean(np.array(data_plot[1]),axis=1) + np.std(np.array(data_plot[1]),axis=1),
                 color='blue', alpha=0.2,label='standard deviation')
	plt.xticks(data_plot[0],labels = data_plot[2])
	plt.ylabel(y_label)
	plt.xlabel(x_label)
	linearRegression = scipy.stats.linregress(np.array(list(data_plot[0])*len(data_plot[1][0])),np.array(data_plot[1]).reshape(  len(data_plot[1])*len(data_plot[1][0])  ) )
	k_t = stats.kendalltau(np.array(list(data_plot[0])*len(data_plot[1][0])),np.array(data_plot[1]).reshape(  len(data_plot[1])*len(data_plot[1][0])  ) )
	plt.plot(data_plot[0],linearRegression.intercept + np.array(data_plot[0])*linearRegression.slope, label='slope={:.3f} \nR2 = {:.3E}\nkt_statistic = {:.3f}\n kt_pvalue = {:.3E}'.format(linearRegression.slope,linearRegression.rvalue**2,k_t[0],k_t[1]),color=color2)
	plt.legend()
	plt.savefig(name + '.pdf')
	plt.close()
	
	file = open(y_label+'/relevant_information_'+ y_label +'.txt','a')
	file.write('R^2 for linear regression between '+ x_label+' and ' + y_label + ': \t'+str(linearRegression.rvalue**2) +'\n')
	file.close()
	
plot_correlation(frequency_common_words_reescaled,colors_luminosity.T,'luminosity/' + 'frequencyCommonWVrLuminosity','luminosity','frequency_common_words','blue','green')
plot_correlation(frequency_common_words_reescaled,colors_chroma.T,'chroma/frequencyCommonWVrChroma','chroma','frequency_common_words','blue','green')
plot_correlation(frequency_common_words_reescaled,colors_hue.T,'hue/frequencyCommonWVrHue','hue','frequency_common_words','blue','green')


plot_correlation(frequency_letters_vocabulary_reescaled,colors_luminosity.T,'luminosity/' + 'frequencyLettersVocabularyVrLuminosity','luminosity','frequency_letters_vocabulary','blue','green')
plot_correlation(frequency_letters_vocabulary_reescaled,colors_chroma.T,'chroma/frequencyLettersVocabularyVrChroma','chroma','frequency_letters_vocabulary','blue','green')
plot_correlation(frequency_letters_vocabulary_reescaled,colors_hue.T,'hue/frequencyLettersVocabularyVrHue','hue','frequency_letters_vocabulary','blue','green')

plot_correlation(frequency_BNC_reescaled,colors_luminosity.T,'luminosity/' + 'frequencyBNCVrLuminosity','luminosity','frequency_BNC','blue','green')
plot_correlation(frequency_BNC_reescaled,colors_chroma.T,'chroma/frequencyBNCVrChroma','chroma','frequency_BNC','blue','green')
plot_correlation(frequency_BNC_reescaled,colors_hue.T,'hue/frequencyBNCVrHue','hue','frequency_BNC','blue','green')

plot_correlation(frequency_picture_books_reescaled,colors_luminosity.T,'luminosity/' + 'frequencyPictureVrLuminosity','luminosity','frequency_picture_books','blue','green')
plot_correlation(frequency_picture_books_reescaled,colors_chroma.T,'chroma/frequencyPictureVrChroma','chroma','frequency_picture_books','blue','green')
plot_correlation(frequency_picture_books_reescaled,colors_hue.T,'hue/frequencyPictureVrHue','hue','frequency_picture_books','blue','green')


######################################################################################################################
#A diferent mesure of correlation
######################################################################################################################
def independent_correlation(colors_,frequency,folder,frequency_name):
	dot=0
	for person in colors_:
		person_reescaled = person/np.linalg.norm(person)
		dot += np.dot(person_reescaled,frequency)
	dot = dot/len(colors_)
	file = open(folder+'/relevant_information_'+ folder +'.txt','a')
	file.write('mean dot product between the ' + str(folder) + ' of each person and the' +frequency_name +': ' + str(dot) +'\n')
	file.close()
	
independent_correlation(colors_chroma,frequency_common_words_reescaled,'chroma','frequency_common_words_reescaled')
independent_correlation(colors_chroma,frequency_letters_vocabulary_reescaled,'chroma','frequency_letters_vocabulary_reescaled')
independent_correlation(colors_chroma,frequency_BNC_reescaled,'chroma','frequency_BNC_reescaled')
independent_correlation(colors_chroma,frequency_picture_books_reescaled,'chroma','frequency_picture_books_reescaled')

independent_correlation(colors_hue,frequency_common_words_reescaled,'hue','frequency_common_words_reescaled')
independent_correlation(colors_hue,frequency_letters_vocabulary_reescaled,'hue','frequency_letters_vocabulary_reescaled')
independent_correlation(colors_hue,frequency_BNC_reescaled,'hue','frequency_BNC_reescaled')
independent_correlation(colors_hue,frequency_picture_books_reescaled,'hue','frequency_picture_books_reescaled')

independent_correlation(colors_luminosity,frequency_common_words_reescaled,'luminosity','frequency_common_words_reescaled')
independent_correlation(colors_luminosity,frequency_letters_vocabulary_reescaled,'luminosity','frequency_letters_vocabulary_reescaled')
independent_correlation(colors_luminosity,frequency_BNC_reescaled,'luminosity','frequency_BNC_reescaled')
independent_correlation(colors_luminosity,frequency_picture_books_reescaled,'luminosity','frequency_picture_books_reescaled')

######################################################################################################################
#compute the natural order acording to the mean, of the letters. 
######################################################################################################################



def lettersOrdered(colors_list,name,y_label,color):
	data_plot = list(zip(colors_list,str_colors,np.mean(colors_list.T,axis=0)))
	data_plot.sort(key = lambda val: val[2])
	data_plot = [[x[0] for x in data_plot],[x[1] for x in data_plot]]
	
	
	#plt.plot(data_plot[0],np.mean(np.array(data_plot[1]),axis=1),color = 'black')
	
	plt.plot(np.arange(26),np.mean(np.array(data_plot[0]),axis=1),color = color)
	plt.fill_between(np.arange(26), np.mean(np.array(data_plot[0]),axis=1) - np.std(np.array(data_plot[0]),axis=1), np.mean(np.array(data_plot[0]),axis=1) + np.std(np.array(data_plot[0]),axis=1)
	                 , alpha=0.2,color=color)
	plt.xticks(np.arange(26),labels = data_plot[1])
	plt.ylabel(y_label)
	plt.savefig(name+'Ordered.pdf')
	plt.close()
	
	
lettersOrdered(colors_chroma.T,'chroma/Chroma','Chroma','green')
lettersOrdered(colors_hue.T,'hue/Hue','Hue','green')
lettersOrdered(colors_luminosity.T,'luminosity/'+'Luminosity','Luminosity','green')



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

#####################################################################################
#Studing the correlation matrix for the lonimo
#####################################################################################




def coor_mat_and_closters(folder,colors_):
	file = open(folder + '/relevant_information_'+folder+'.txt','a')
	corr_mat = Corr_Mat(colors_)
	corr_mat_Sh = np.zeros(26*26*2)
	for i in tqdm(range(2)):
		colorsSh = shuffleL(colors_)
		#print('computing covariance matrix of shuffle data {}...'.format(i))
		corr_mat_Sh[26*26*i:26*26*(i+1)] = Corr_Mat(colorsSh.T)
	plot_histogram(corr_mat,60,'blue',label = 'original data')
	plot_histogram(corr_mat_Sh,60,'green',label='shuffle data')
	test = stats.kstest(corr_mat,corr_mat_Sh)
	pV = test[1]
	plt.text(0.8,-plt.ylim()[1]/10,'p_value = %.2E'%pV)
	plt.legend()
	plt.savefig(folder + '/corr_histVrsShuffle.pdf')
	plt.close()
	
	corr_mat_Sh = corr_mat_Sh.reshape(2,26*26)
	mean_corr_mat_Sh = np.mean(corr_mat_Sh,axis=0)
	
	plotEigenvalues(mean_corr_mat_Sh,folder+'/eigenvaluesSh.pdf','eigenvalues of shuffle data')
	corr_mat_eiva , corr_mat_eive, index = plotEigenvalues(corr_mat,folder+'/eigenvalues.pdf','eigenvalues',cap='True')
	corr_mat_eive_reescaled = corr_mat_eive/np.linalg.norm(corr_mat_eive,axis=0)
	
	
	corr_mat = corr_mat.reshape((26,26))
	cmap = cm.get_cmap('plasma', 256)
	psm = plt.pcolormesh(corr_mat, cmap=cmap, rasterized=True, vmin=np.min(corr_mat), vmax=np.max(corr_mat))
	plt.colorbar(psm)
	plt.savefig(folder+'/corr_mat' + '.pdf')
	plt.close()
	Ofile = open(folder +'/corr_mat.txt','w')
	for col in corr_mat:
		for element in col:
			Ofile.write(str(element)+' ')
		Ofile.write('/n')
	Ofile.close()
	
	
	file.write('dot product frequency_picture_books and corr_mat_eive: \t' +  str(np.dot(frequency_picture_books_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))  + '\n' )
	file.write('dot product frequency_handwriting_books and corr_mat_eive: \t' + str(np.dot(frequency_handwriting_books_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))+ '\n' )
	file.write('dot product frequency_common_words and corr_mat_eive: \t' + str(np.dot(frequency_common_words_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))+ '\n' )
	file.write('dot product frequency_letters_vocabulary and corr_mat_eive: \t' + str(np.dot(frequency_letters_vocabulary_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))+ '\n' )
	file.write('dot product max eigenvalue of magnet toy and corr_mat_eive: \t' + str(np.dot(eigen_magnets[1][np.argmax(eigen_magnets[0])],corr_mat_eive[np.argmax(index)]))+ '\n' )
	
	
	
	#compute the pairwise distances between points in Lab space
	distances = pdist(colors_)
	
	#make the aglomerative ward linkage clostering. Outcome is codified. #This was the one that permormed the best.
	
	closters_linkage = linkage(distances,method='ward',metric='euclidean')
	
	#let's see to what closter, each point belong. 
	
	closter_indexing = fcluster(closters_linkage,9,criterion = 'maxclust')
	
	#storing the clostered data
	closters = []
	for i in range(np.max(closter_indexing)):
		closters.append(colors[closter_indexing==i+1])
		
	
	for j in tqdm(range(np.max(closter_indexing))):
		N = len(closters[j])
		file.write('lenght of the cluster: \t' + str(N) + '\n')
		
		#independent_correlation(closters[j],frequency_common_words_reescaled,folder,'frequency_common_words_reescaled')
		#independent_correlation(closters[j],frequency_letters_vocabulary_reescaled,folder,'frequency_letters_vocabulary_reescaled')
		#independent_correlation(closters[j],frequency_BNC_reescaled,folder,'frequency_BNC_reescaled')
		#independent_correlation(closters[j],frequency_picture_books_reescaled,folder,'frequency_picture_books_reescaled')
		
		corr_mat = Corr_Mat(closters[j].reshape((N,3,26)).T)
		corr_mat_Sh = np.zeros(26*26*10)
		for i in range(10):
			colorsSh = shuffleL(closters[j])
			#print('computing covariance matrix of shuffle data {}...'.format(i))
			corr_mat_Sh[26*26*i:26*26*(i+1)] = Corr_Mat(colorsSh.T)
		plot_histogram(corr_mat,60,'blue',label = 'original data')
		plot_histogram(corr_mat_Sh,60,'green',label='shuffle data')
		test = stats.kstest(corr_mat,corr_mat_Sh)
		pV = test[1]
		plt.text(0.8,-plt.ylim()[1]/10,'p_value = %.2E'%pV)
		plt.legend()
		plt.savefig(folder+'/corr_histVrsShuffle_closter{}.pdf'.format(j+1))
		plt.close()
		
		corr_mat_Sh = corr_mat_Sh.reshape(10,26*26)
		mean_corr_mat_Sh = np.mean(corr_mat_Sh,axis=0)
	
	
		plotEigenvalues(mean_corr_mat_Sh,folder+'/eigenvaluesSh_closter{}.pdf'.format(j+1),'eigenvalues of shuffle data')
		corr_mat_eiva , corr_mat_eive, index = plotEigenvalues(corr_mat,folder+'/eigenvalues_closter{}.pdf'.format(j+1),'eigenvalues',cap='True')
		corr_mat_eive_reescaled = corr_mat_eive/np.linalg.norm(corr_mat_eive,axis=0)
		
		file.write('dot product frequency_picture_books and corr_mat_eive closter {}: \t'.format(j) +  str(np.dot(frequency_picture_books_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))  + '\n' )
		file.write('dot product frequency_handwriting_books and corr_mat_eive closter {}: \t'.format(j) +  str(np.dot(frequency_handwriting_books_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))+ '\n' )
		file.write('dot product frequency_common_words and corr_mat_eive closter {}: \t'.format(j) +  str(np.dot(frequency_common_words_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))+ '\n' )
		file.write('dot product frequency_letters_vocabulary and corr_mat_eive closter {}: \t'.format(j) +  str(np.dot(frequency_letters_vocabulary_reescaled,corr_mat_eive_reescaled[np.argmax(index)]))+ '\n' )
		file.write('dot product max eigenvalue of magnet toy and corr_mat_eive closter {}: \t'.format(j) +  str(np.dot(eigen_magnets[1][np.argmax(eigen_magnets[0])],corr_mat_eive[np.argmax(index)]))+ '\n' )
	file.close()

		
coor_mat_and_closters('luminosity',colors_luminosity)
coor_mat_and_closters('hue',colors_hue)
coor_mat_and_closters('chroma',colors_chroma)






