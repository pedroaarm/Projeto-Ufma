import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn import preprocessing as pp
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import salva_resultados as sr
from os import walk
import os

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
path="C:/Users/danil/Documents/ProjetoUfma/Dados/treino_SVM_testes"
pathResults="C:/Users/danil/Documents/ProjetoUfma/Dados/treino_SVM_testes/Resultados"


#metodo para calcular a acuracia utilizando cross validation
def accuracy(clf,X,y,fold):

   resultados = cross_val_predict(clf, X, y, cv=fold) 
   result=accuracy_score(y,resultados)
   return result  	

#metodo que executa um simples grid search variando kernels,parametro c e parametro gamma. Salva os resultados ao final
def gridSearch(dataSet,label,faixa,dest,index):

	output=[]
	c_list=[1,32,64,128,512,1024] #lista com os parametros C
	gamma_list=[1,32,64,128,512,1024] #lista com os parametros gamma 
	kernels=['linear','rbf','poly'] #lista com os kernels
	folders=[3,5,7,9]

	for kern in kernels:
		if(kern=='linear'):
			for c in c_list:
				for folder in folders:	
					model=svm.SVC(C=c,kernel=kern) #cria o modelo de aprendizado
					model.fit(dataSet,label)
					result=accuracy(model,dataSet,label,folder)
					output.append(sr.FormatData(c,0,round(result,3),folder,faixa,kern))
		else:
			for c in c_list:
				for g in gamma_list:
					for folder in folders:	
						model=svm.SVC(C=c,kernel=kern,gamma=g) #cria o modelo de aprendizado
						model.fit(dataSet,label)
						result=accuracy(model,dataSet,label,folder)
						output.append(sr.FormatData(c,g,round(result,3),folder,faixa,kern))
	
	dest=dest+"resultadoClassificadorFeatures"+str(index)+".csv" # gerenado um arquivo .csv onde os resultados serão salvos

	if(os.path.exists(dest)!=True):
		arq=open(dest,"w")
		arq.write("faixa;kernel;c;gamma;resultado;folders\n")
		arq.close()

	sr.FormatData.saveData(dest,output)				
					
def testClassifier(path):

	#Este metodo testa varios hiperparametros para um aprendizado de maquina utilizando SVM por validaçao cruzada(cross-validation).
	#Utiliza 7 faixas de discretização das temperaturas, 3 conjuntos de features diferentes, 4 metodos para padronizar os dados, 6 valores para o parametro C,
	#6 valores para o parametro gamma, 3 kernels diferentes e 4 valores para os folders do cross-validation.
	#São gerados 36.288 resultados.

	featureList=[
	['homogenity1variance','homogenity1kurt','homogenity1std','energy2skew','homogenity2variance','dissimilarity2variance'],
	['homogenity1variance','homogenity1kurt','energy2skew','dissimilarity2variance'],
	['energy1kurt','energy1skew','entropy1variance','homogenity1variance','homogenity1std','homogenity2variance','homogenity2std','correlation2mean','dissimilarity2variance','dissimilarity2std']]

	scalerList=[1,2,3,4]

	global pathResults

	for raiz,diretorios,arquivos in walk(path):
	
		for i in range(0,len(diretorios)):

			subpath=path+diretorios[i]
			print("Lendo diretorio {0}".format(diretorios[i]))

			for sub_dir_path,sub_dir,sub_dir_arquivos in walk(subpath):
				l=1
				for j in range(0,len(sub_dir_arquivos)):

					if(sub_dir_arquivos[j].find("aprendizado"))!=-1:

						origem=subpath+'/'+sub_dir_arquivos[j]
						aprendizado=pd.read_csv(origem)

						for feat in featureList:
							
							#esta parte do código padroniza os dados com 4 scalers diferentes e computa o gridSerch para os dados padronizados em cada um dos Scalers escolhidos
							scaler=pp.MinMaxScaler(feature_range=(0,1)).fit(aprendizado[feat])
							aprendizado[feat]=scaler.transform(aprendizado[feat])
							dest=pathResults+"MinMaxScaler/"
							pacientes=aprendizado[feat].values
							type_label = np.where(aprendizado['class'] == 'healthy',0,1)
							gridSearch(pacientes,type_label,diretorios[i],dest,l)

							scaler=pp.StandardScaler().fit(aprendizado[feat])
							aprendizado[feat]=scaler.transform(aprendizado[feat])
							dest=pathResults+"StandardScaler/"
							pacientes=aprendizado[feat].values
							type_label = np.where(aprendizado['class'] == 'healthy',0,1)
							gridSearch(pacientes,type_label,diretorios[i],dest,l)

							scaler=pp.RobustScaler().fit(aprendizado[feat])
							aprendizado[feat]=scaler.transform(aprendizado[feat])
							dest=pathResults+"RobustScaler/"
							pacientes=aprendizado[feat].values
							type_label = np.where(aprendizado['class'] == 'healthy',0,1)
							gridSearch(pacientes,type_label,diretorios[i],dest,l)

							scaler=pp.Normalizer().fit(aprendizado[feat])
							aprendizado[feat]=scaler.transform(aprendizado[feat])
							dest=pathResults+"Normalizer/"
							pacientes=aprendizado[feat].values
							type_label = np.where(aprendizado['class'] == 'healthy',0,1)
							gridSearch(pacientes,type_label,diretorios[i],dest,l)
			
							l+=1	#variavel para diferenciar qual conjunto de features esta sendo usado
						
					else:
						break

				break	

def main():

	global path

	testClassifier(path)

main()