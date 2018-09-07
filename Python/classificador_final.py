import pandas as pd 
import numpy as np
import sklearn
from sklearn import svm
from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score
import salva_resultados as sr
import os
from sklearn.metrics import confusion_matrix

#feat=['homogenity1variance','homogenity1kurt','homogenity1std','energy2skew','homogenity2variance','dissimilarity2variance']
#feat=['energy1mean','energy1variance','energy1kurt','energy1skew','energy1std','entropy1mean','entropy1variance','entropy1kurt','entropy1skew','entropy1std','contrast1mean','contrast1variance','contrast1kurt','contrast1skew','contrast1std','homogenity1mean','homogenity1variance','homogenity1kurt','homogenity1skew','homogenity1std','correlation1mean','correlation1variance','correlation1kurt','correlation1skew','correlation1std','dissimilarity1mean','dissimilarity1variance','dissimilarity1kurt','dissimilarity1skew','dissimilarity1std','energy2mean','energy2variance','energy2kurt','energy2skew','energy2std','entropy2mean','entropy2variance','entropy2kurt','entropy2skew','entropy2std','contrast2mean','contrast2variance','contrast2kurt','contrast2skew','contrast2std','homogenity2mean','homogenity2variance','homogenity2kurt','homogenity2skew','homogenity2std','correlation2mean','correlation2variance','correlation2kurt','correlation2skew','correlation2std','dissimilarity2mean','dissimilarity2variance','dissimilarity2kurt','dissimilarity2skew','dissimilarity2std']
feat=['energy1kurt','energy1skew','entropy1variance','homogenity1variance','homogenity1std','homogenity2variance','homogenity2std','correlation2mean','dissimilarity2variance','dissimilarity2std']

treinoPath="C:/Users/danil/Documents/ProjetoUfma/Dados/TesteFinal/aprendizado.csv"
testePath="C:/Users/danil/Documents/ProjetoUfma/Dados/TesteFinal/teste.csv"
validaPath="C:/Users/danil/Documents/ProjetoUfma/Dados/TesteFinal/valida.csv"
myPath="C:/Users/danil/Documents/ProjetoUfma/Dados/TesteFinal/testeFinal.csv"


def avaliaçao(cnf_matrix):

    vp = cnf_matrix[0][0] # VP
    fn = cnf_matrix[0][1] # FN
    fp = cnf_matrix[1][0] # FP
    vn = cnf_matrix[1][1] # VN

    acuracia = (vp+vn)/(vp+vn+fp+fn) # perc de acertos total
    sensibilidade = vp/(vp+fn) # perc de positivos acertados # recall
    especificidade = vn/(vn+fp) # perc de negativos acertados
    precisao = vp/(vp+fp) # prob de repetir a acuracia

   
    print ("acuracia = ", acuracia)
    print ("sensibilidade = ", sensibilidade)
    print ("especificidade = ", especificidade)
    print ("precisao = ", precisao)

    #return acuracia, sensibilidade


def createModel(path):

	global feat

	model=svm.SVC(C=0.01,kernel='rbf',gamma=0.01)
	
	data=pd.read_csv(path)

	scaler=pp.MinMaxScaler(feature_range=(0,1)).fit(data[feat])
	data[feat]=scaler.transform(data[feat])
			
	treino=data[feat].values
	labels=np.where(data['class']=='healthy',0,1)	
	model.fit(treino,labels)

	return model		


def createData(path):
	
	global feat	
	data=pd.read_csv(path)
	
	scaler=pp.MinMaxScaler(feature_range=(0,1)).fit(data[feat])

	data[feat]=scaler.transform(data[feat])

	output=data[feat].values

	return output	

		
def classify(model,data,path,size):

	result=[]
	file=pd.read_csv(path)

	labels=np.where(file['class']=='healthy',0,1)

	for val in data:
		result.append(model.predict([val]))

	result=np.asarray(result)	
	result=np.reshape(result,(size)) # 12 para o teste, 14 para a validaçao
	accuracy=accuracy_score(labels,result)

	cnf_matrix=confusion_matrix(labels,result)
	avaliaçao(cnf_matrix)

	return accuracy



def runSVM(trainPath,thePath):
	


	output=[]
	dest="C:/Users/danil/Documents/ProjetoUfma/Dados/TesteFinal/meuclassificadorSVM.csv"

	if(os.path.exists(dest)!=True):
		arq=open(dest,"w")
		arq.write("faixa;kernel;c;padronizador;resultado;test_val\n")
		arq.close()

	svmModel=createModel(trainPath)

	#data1=createData(testPath)
	#data2=createData(validaPath)
	data=createData(myPath)
	'''
	print("Classificando o grupo de testes")
	results1=classify(svmModel,data1,testPath,12)
	print("Classificando o grupo de validação")
	results2=classify(svmModel,data2,validaPath,14)
	'''
	print("Calssificando")
	results=classify(svmModel,data,myPath,26)
	#output.append(sr.FormatData(0.01,round(results1,3),1,'0.20','rbf','teste'))
	#output.append(sr.FormatData(0.01,round(results2,3),1,'0.20','rbf','valida'))
	
	#sr.FormatData.saveData(dest,output)


def main():

	global treinoPath
	global testePath
	global validaPath
	global myPath

	runSVM(treinoPath,myPath)


main()	