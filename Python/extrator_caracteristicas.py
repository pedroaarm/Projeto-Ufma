import numpy as np
import images
from math import floor,floor,pow,log2,sqrt,fabs

#código com todos os metodos para extrair as caracteristicas selecionadas

def energy(mat):

	mat=mat**2
	energy=np.sum(mat)
	
	return energy

def entropy(mat):

	entropy=0
	for i in range(0,mat.shape[0]):
		for j in range(0,mat.shape[1]):
			if(mat[i][j]!=0):
				entropy+=mat[i][j]*log2(mat[i][j])

	return entropy*(-1)

def contrast(mat):

	contrast=0
	for i in range(0,mat.shape[0]):
		for j in range(0,mat.shape[1]):
			contrast+=pow((i-j),2)*mat[i][j]

	return contrast

def homogenity(mat):

	homogenity=0
	for i in range(0,mat.shape[0]):
		for j in range(0,mat.shape[1]):
			homogenity+=(mat[i][j])/(1+fabs(i-j))

	return homogenity

def correlation(mat):

	mI=0
	mJ=0

	#extraindo medias das linhas I e colunas J
	for i in range(0,mat.shape[0]):
		for j in range(0,mat.shape[1]):
			mI+=i*(mat[i][j])

	for j in range(0,mat.shape[0]):
		for i in range(0,mat.shape[1]):
			mJ+=j*(mat[i][j])

	stdI=0
	stdJ=0

	#extraindo desvios padrao para linhas I e colunas J
	for i in range(0,mat.shape[0]):
		for j in range(0,mat.shape[1]):
			stdI+=pow((i-mI),2)*(mat[i][j])

	for j in range(0,mat.shape[0]):
		for i in range(0,mat.shape[1]):
			stdJ+=pow((j-mJ),2)*(mat[i][j])

	stdI=sqrt(stdI)
	stdJ=sqrt(stdJ)

	#calculando a correlação
	correlation=0
	for i in range(0,mat.shape[0]):
		for j in range(0,mat.shape[1]):		
			correlation+=(i-mI)*(j-mJ)*mat[i][j]

	return (correlation/(stdI*stdJ))


def dissimilarity(mat):

	dissimilarity=0

	for i in range(0,mat.shape[0]):
		for j in range(0,mat.shape[1]):
			dissimilarity+=fabs(i-j)*mat[i][j]

	return dissimilarity
	
'''
Este método recebe como parâmetros uma matriz de temperaturas,uma distancia e um ângulo entre(0,90°,180° ou 270°)
e calcula a matriz de co-ocorrência de temperaturas para a dada entrada.
'''
def getCoMatrix(mat,distance,angle):

	coMatrix=np.zeros((45,45)) #faixa de valores variando de 0.20.

	for i in range(0,mat.shape[0]):
		for j in range(0,mat.shape[1]):
			try:
				if angle==0:

					if(mat[i][j]!=0):
						row=floor(((mat[i][j]-27)*44)/(9))                     
						if(mat[i][j+distance]!=0):
							col=floor(((mat[i][j+distance]-27)*44)/(9))
						else:
							raise ValueError
					else:
						raise ValueError

				elif angle==(3*np.pi)/2:

					if(mat[i][j]!=0):
						row=floor(((mat[i][j]-27)*44)/(9))                     
						if(mat[i+distance][j]!=0):
							col=floor(((mat[i+distance][j]-27)*44)/(9))
						else:
							raise ValueError
					else:
						raise ValueError

				elif angle==np.pi:

					if(j-distance<0):
						raise IndexError

					if(mat[i][j]!=0):
						row=floor(((mat[i][j]-27)*44)/(9))                     
						if(mat[i][j-distance]!=0):
							col=floor(((mat[i][j-distance]-27)*44)/(9))
						else:
							raise ValueError
					else:
						raise ValueError

				elif angle==np.pi/2:
					
					if(i-distance<0):
						raise IndexError 
					
					if(mat[i][j]!=0):
						row=floor(((mat[i][j]-27)*44)/(9))                     
						if(mat[i-distance][j]!=0):
							col=floor(((mat[i-distance][j]-27)*44)/(9))
						else:
							raise ValueError
					else:
						raise ValueError

				else:
					raise ValueError	

			except IndexError: 
				pass
			except ValueError:
				pass
			else:
				coMatrix[row][col]+=1

	#normalizando matriz
	soma=np.sum(coMatrix)
	coMatrix/=soma

	return coMatrix

def extractGreyCoProps(coMatrix):
	
	output=[]
	output.append(energy(coMatrix))
	output.append(entropy(coMatrix))
	output.append(contrast(coMatrix))
	output.append(homogenity(coMatrix))
	output.append(correlation(coMatrix))
	output.append(dissimilarity(coMatrix))

	return output

def mean(vec): #funçao para calcular a media simples
	
	soma=0
	for i in range(0,len(vec)):
		soma+=vec[i]

	return soma/20

def variance(vec): #funçao para calcular a variancia
	
	
	
	mn=mean(vec)
	mySum=0
	
	for i in range(0,len(vec)):
		mySum+=pow((vec[i]-mn),2)
	
	var=(mySum)/(20)
	#print(var)
	

	return round(var,9)

def skewness(vec): #funçao para calcular a obliquidade
	
	auxMean=(np.sum(vec))/20
	auxSum=0
	auxN=(20)
	var=variance(vec)

	for i in range(0,len(vec)):
		auxSum+=(pow((vec[i]-auxMean),3))

	auxSum=(auxSum/20)
	skew=(((sqrt((auxN)*(auxN-1)))/(auxN-2))*(auxSum/(pow(sqrt(var),3))))
	#print(skew)
	return round(skew,8)

def kurtosis(vec): #funçaõ para calcular a curtose
	
	auxMean=(np.sum(vec))/20
	auxSum=0
	var=variance(vec)

	for i in range(0,len(vec)):
		auxSum+=(pow((vec[i]-auxMean),4))

	auxSum=(auxSum/20)
	kurt=(auxSum/(pow(sqrt(var),4)))
	#adjust=((20-1)*((20+1)*kurt+7))/((20-2)*(20-3));
	#kurt*=adjust
	#print(kurt)
	return round((kurt-3),8)

def standardDeviation(vec): #função para calcular o desvio padrao
	
	std=variance(vec)

	return round(sqrt(std),8)


def computeCharacteristics(matrix):

	#função para computar todas as caracteristicas selecionadas e adiciona-las a
	#um vetor de caracteristicas 
	output=[]

	#extraindo matrizes de co-ocorrencias
	coMatrix1=getCoMatrix(matrix,1,0)
	coMatrix2=getCoMatrix(matrix,1,(3*np.pi)/2)
	#extraindo todos os descritores de textura das matrizes computadas
	output.extend(extractGreyCoProps(coMatrix1))
	output.extend(extractGreyCoProps(coMatrix2))

	return output
	

def extractStatistics(images):

	index=0
	auxVec=[]
	i=0
	output=[]

	while i<12:

		for j in range(0,len(images)):
			aux=images[j].GetVector()
			auxVec.append(aux[i])
		
		output.append(round(mean(auxVec),3))
		output.append(round(variance(auxVec),3))
		output.append(round(kurtosis(auxVec),3))
		output.append(round(skewness(auxVec),3))
		output.append(round(standardDeviation(auxVec),3))
		
		auxVec.clear()

		i+=1

	return output	

