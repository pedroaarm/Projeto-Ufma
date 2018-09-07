import pandas as pd 
import numpy as np
import sklearn
from sklearn import svm
from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score
import salva_resultados as sr
from os import walk
import os


#MinMax e Standard Scaler tiveram melhores resultados

'''
#Scalers
#StandardScaler----formula----xi–mean(x)/stdev(x)
#MinMaxScaler(perdem eficiencia com a presença de outliers)----formula----xi–min(x)/max(x)–min(x)
#RobustScaler----formula----xi–Q1(x)/Q3(x)–Q1(x)
#Normalizer----formula----xi/√xi^2+yi^2+zi^2

#feature1=['homogenity1variance','homogenity1kurt','homogenity1std','energy2skew','homogenity2variance','dissimilarity2variance']
#feature2=['homogenity1variance','homogenity1kurt','energy2skew','dissimilarity2variance']
#feature3=['energy1kurt','energy1skew','entropy1variance','homogenity1variance','homogenity1std','homogenity2variance','homogenity2std','correlation2mean','dissimilarity2variance','dissimilarity2std']

#features --- >'energy1mean','energy1variance','energy1kurt','energy1skew','energy1std','entropy1mean','entropy1variance','entropy1kurt','entropy1skew','entropy1std','contrast1mean','contrast1variance','contrast1kurt','contrast1skew','contrast1std','homogenity1mean','homogenity1variance','homogenity1kurt','homogenity1skew','homogenity1std','correlation1mean','correlation1variance','correlation1kurt','correlation1skew','correlation1std','dissimilarity1mean','dissimilarity1variance','dissimilarity1kurt','dissimilarity1skew','dissimilarity1std','energy2mean','energy2variance','energy2kurt','energy2skew','energy2std','entropy2mean','entropy2variance','entropy2kurt','entropy2skew','entropy2std','contrast2mean','contrast2variance','contrast2kurt','contrast2skew','contrast2std','homogenity2mean','homogenity2variance','homogenity2kurt','homogenity2skew','homogenity2std','correlation2mean','correlation2variance','correlation2kurt','correlation2skew','correlation2std','dissimilarity2mean','dissimilarity2variance','dissimilarity2kurt','dissimilarity2skew','dissimilarity2std'"
'''

treinoPath="C:/Users/danil/Documents/ProjetoUfma/Dados/treino_SVM_definitivo/Faixa_020/aprendizado.csv"
testePath="C:/Users/danil/Documents/ProjetoUfma/Dados/treino_SVM_definitivo/Faixa_020/teste.csv"

def createModel(path,c,kern,g,scaler,feat):

	if kern=='linear':
		model=svm.SVC(C=c,kernel=kern)
	else:
		model=svm.SVC(C=c,kernel=kern,gamma=g)
	
	data=pd.read_csv(path)

	if scaler==1:
		scaler=pp.MinMaxScaler(feature_range=(0,1)).fit(data[feat])
		data[feat]=scaler.transform(data[feat])
	elif scaler==2:
		scaler=pp.StandardScaler().fit(data[feat])
		data[feat]=scaler.transform(data[feat])
			
	treino=data[feat].values
	labels=np.where(data['class']=='healthy',0,1)	
	model.fit(treino,labels)

	return model		

def createData(path,scaler,feat):
	
	data=pd.read_csv(path)
	if scaler==1:
		scaler=pp.MinMaxScaler(feature_range=(0,1)).fit(data[feat])
	elif scaler==2:
		scaler=pp.StandardScaler().fit(data[feat])
	
	data[feat]=scaler.transform(data[feat])

	output=data[feat].values


	return output	
		
def classify(model,data,path):

	result=[]
	file=pd.read_csv(path)

	labels=np.where(file['class']=='healthy',0,1)

	for val in data:
		result.append(model.predict([val]))

	result=np.asarray(result)	
	result=np.reshape(result,(12)) # 12 para o teste, 14 para a validaçao
	accuracy=accuracy_score(labels,result)

	return accuracy

def runSVM(trainPath,testPath):
	
	features=['homogenity1variance','homogenity1kurt','homogenity1std','energy2skew','homogenity2variance','dissimilarity2variance']
	kernels=['linear','rbf']
	c_list=[1,32,64,128,512,1024]
	gamma_list=[1,32,64,128,512,1024]
	scalers=[1,2]
	output=[]
	dest="C:/Users/danil/Documents/ProjetoUfma/Dados/treino_SVM_definitivo/Faixa_005/Resultados/validaçaoSVM.csv"

	if(os.path.exists(dest)!=True):
		arq=open(dest,"w")
		arq.write("faixa;kernel;c;gamma;padronizador;resultado\n")
		arq.close()

	for kern in kernels:

		if kern=='linear':
			for c in c_list:
				for scal in scalers:
					svmModel=createModel(trainPath,c,kern,0,scal,features)
					data=createData(testPath,scal,features)
					results=classify(svmModel,data,testPath)
					output.append(sr.FormatData(c,0,round(results,3),scal,'0.05',kern))
	
		else:
			for c in c_list:
				for g in gamma_list:
					for scal in scalers:
						svmModel=createModel(trainPath,c,kern,g,scal,features)
						data=createData(testPath,scal,features)
						results=classify(svmModel,data,testPath)
						output.append(sr.FormatData(c,g,round(results,3),scal,'0.05',kern))
	
	#sr.FormatData.saveData(dest,output)	

def main():

	global treinoPath
	global testePath

	runSVM(treinoPath,testePath)



main()	
