import os
import numpy as np
import images
import extrator_caracteristicas as ec
from os import walk
import paciente
import gerar_arquivo_csv as gcsv
import random


'''
código que carrega toda a base de dados e extrai todas as matrizes de temperatura.
apos extrai todas as caracteristicas selecionadas de todo o conjunto de imagens de cada paciente.
ao final salva todas informações extraidas em um arquivo do tipo .csv
'''


arquivo1="C:/Users/danil/Documents/ProjetoUfma/Dados/TesteFinal/aprendizado.csv"
arquivo2="C:/Users/danil/Documents/ProjetoUfma/Dados/TesteFinal/teste.csv"
arquivo3="C:/Users/danil/Documents/ProjetoUfma/Dados/TesteFinal/valida.csv"
	
atributos="class,energy1mean,energy1variance,energy1kurt,energy1skew,energy1std,entropy1mean,entropy1variance,entropy1kurt,entropy1skew,entropy1std,contrast1mean,contrast1variance,contrast1kurt,contrast1skew,contrast1std,homogenity1mean,homogenity1variance,homogenity1kurt,homogenity1skew,homogenity1std,correlation1mean,correlation1variance,correlation1kurt,correlation1skew,correlation1std,dissimilarity1mean,dissimilarity1variance,dissimilarity1kurt,dissimilarity1skew,dissimilarity1std,energy2mean,energy2variance,energy2kurt,energy2skew,energy2std,entropy2mean,entropy2variance,entropy2kurt,entropy2skew,entropy2std,contrast2mean,contrast2variance,contrast2kurt,contrast2skew,contrast2std,homogenity2mean,homogenity2variance,homogenity2kurt,homogenity2skew,homogenity2std,correlation2mean,correlation2variance,correlation2kurt,correlation2skew,correlation2std,dissimilarity2mean,dissimilarity2variance,dissimilarity2kurt,dissimilarity2skew,dissimilarity2std"

arq=open(arquivo1,'w')
arq.write("{0}{1}".format(atributos,'\n'))
arq.close()
arq=open(arquivo2,'w')
arq.write("{0}{1}".format(atributos,'\n'))
arq.close()
arq=open(arquivo3,'w')
arq.write("{0}{1}".format(atributos,'\n'))
arq.close()


countSickAprend=0
countHealthyAprend=0
countSickValida=0
countHealthyValida=0

def generateCSVFile(pacientes):

	aprendizado=[] #44
	teste=[] #12
	validaçao=[] #14

	global countSickAprend
	global countHealthyAprend
	global countSickValida
	global countHealthyValida

	global arquivo1
	global arquivo2 
	global arquivo3

	random.shuffle(pacientes)

	for i in range(0,len(pacientes)):
		#cria uma lista contendo os dados para serem salvos no arquivo csv
		#esse arquivo vai ser usado posteriormente como base de dados para os classificadores
		if pacientes[i].GetClasse() == 'sick':
			if(countSickAprend<22):
				aprendizado.append(gcsv.FormatData(pacientes[i].GetName(),pacientes[i].GetClasse(),pacientes[i].GetFeatures()))
				countSickAprend+=1
			elif countSickValida<7:
				validaçao.append(gcsv.FormatData(pacientes[i].GetName(),pacientes[i].GetClasse(),pacientes[i].GetFeatures()))
				countSickValida+=1
			else:
				teste.append(gcsv.FormatData(pacientes[i].GetName(),pacientes[i].GetClasse(),pacientes[i].GetFeatures()))
		
		elif pacientes[i].GetClasse() == 'healthy':
			if(countHealthyAprend<22):
				aprendizado.append(gcsv.FormatData(pacientes[i].GetName(),pacientes[i].GetClasse(),pacientes[i].GetFeatures()))
				countHealthyAprend+=1	
			elif countHealthyValida<7:
				validaçao.append(gcsv.FormatData(pacientes[i].GetName(),pacientes[i].GetClasse(),pacientes[i].GetFeatures()))
				countHealthyValida+=1
			else:
				teste.append(gcsv.FormatData(pacientes[i].GetName(),pacientes[i].GetClasse(),pacientes[i].GetFeatures()))


	gcsv.FormatData.saveData(arquivo1,aprendizado)	
	gcsv.FormatData.saveData(arquivo2,teste)
	gcsv.FormatData.saveData(arquivo3,validaçao)	


def createBoundBox(matrix):
	a=np.where(matrix!=0)
	bbox=matrix[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]

	return bbox

def extract(pacientes):
	
	for i in range(0,len(pacientes)): #varre todos os pacientes e extrai as caracteristicas escolhidas para cada imagem
		 #adquire a lista de imagens
		print("Extraindo de {0}".format(pacientes[i].GetName()))

		output=[]
		files=pacientes[i].GetImages()
		for j in range(0,len(files)):
			output=ec.computeCharacteristics(files[j].GetMatrix()) # seta as caracteristicas para cada imagem
			files[j].SetVector(output)

		pacientes[i].SetFeatures(ec.extractStatistics(pacientes[i].GetImages()))


	return

def loadBase(path):

	pacientes = []

	for(dir_path,dir_names,file_names) in walk(path):#procura todos os subdiretorios no path
		
		for i in range(0,len(dir_names)):
			print('dir names = ',dir_names[i])
			subpath = path+dir_names[i]

			for(sub_dir_path,sub_dir,file_names) in walk(subpath): #subdiretorios doente e saudavel
				#print('lista sub_dir_names = ',sub_dir)
				exam_name = sub_dir
				break

			for j in range(0,len(exam_name)): #cria o path para todas as imagens de cada paciente
				
				exam_path=subpath+'/'+exam_name[j]
				#print('exam path = '+exam_path)
				#nome_do_exame=exam_name[j]
				files=[]
				for(paths,dirs,arq_names) in walk(exam_path):#anda dentro do subdiretorio de cada paciente
					if dir_names[i] == 'doente':
						auxPat=paciente.Patient(exam_name[j],"sick")
					elif dir_names[i] == 'saudavel':
						auxPat=paciente.Patient(exam_name[j],"healthy")
					#auxPat.SetName(exam_name[j])
					#auxPat.SetClasse(dir_names[i])
					for k in range(0,len(arq_names)): # extrai todas as matrizes em .txt 

						if arq_names[k].find(".txt") != -1:
							#print("Arquivo "+arq_names[k])
							auxIm=images.Image(arq_names[k])
							matrix = np.loadtxt(exam_path+'/'+arq_names[k])
							bbox=createBoundBox(matrix)
							auxIm.SetMatrix(bbox)
							files.append(auxIm)

					auxPat.SetImages(files)		
					#files.clear()
					pacientes.append(auxPat)
					#del auxPat
					break
				#print("Adicionando imagens de paciente "+auxPat.GetName())				

			extract(pacientes)
			random.shuffle(pacientes)
			generateCSVFile(pacientes)
			pacientes.clear()
				
		return		

def main():
	
	#variaveis utiilzadas na execução do programa
	origem="C:/Users/danil/Documents/ProjetoUfma/BASE-lincoln-ROI-PNG/" 
	novoDiretorio="C:/Users/danil/Documents/ProjetoUfma/"

	if (os.path.exists(novoDiretorio+"Dados")) != True :
		os.mkdir(novoDiretorio+"Dados")

	loadBase(origem)
	


main()