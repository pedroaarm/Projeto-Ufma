
# ESSE CÓDIGO FAZ: 
# pega o banco de mamas DMR 
# e mascara as matrizes de temperatura, e produz as suas imagens em PNG.
# (tudo em um diretorio separado, da mesma forma que o original.)


'''
PASSOS DETALHADOS:

Percorrer o diretorio da base de origem 
	Percorrer as subbases (doentes, saudável)
		Entrar na pasta de exame
			Pegar a máscara do exame
			Achar todos os txt de temp registrados
			Subtrai pela máscara do exame (tratar limiar de 128 na máscara para eliminar ruidos)
			Salvar as txt de temp (num diretório definido, como uma nova base) ('nomeexame-XX.txt')
			Salvar os txt de temp como imagens png ('nomeexame-XX.png')

'''



# Imports
from PIL import Image
#import cv2
from os import walk
import numpy as np
import os


from PIL import Image
import random
import scipy.misc




# Import folder
def load_dataset(path, dest):
	print ("Gerando imagens em:")
	print (dest)
	print("\n")

	maior = 0
	menor = 1000

	destinofinal = ''
	f = [] # folder    
	dirr = [] # dir from this folder 
	# Find all subfolders in folder
	for (dir_path, dir_names, file_names) in walk(path):
		f.extend(dir_names)
		dirr.extend(dir_names)


		for i in range(0, len(dir_names)): # ENTRA nos subs SAUDAVEL e DOENTE
			print('dir_names = ',dir_names[i])
			subpath = path+'/'+dir_names[i]   # dir + /doente ou /saudavel
			print('subpath = ',subpath)

			destinofinal = dest+'/'+dir_names[i] # comeca a construir o destino
			if not os.path.exists(destinofinal):
				os.makedirs(destinofinal) 

			# Capta os nome DOS EXAMES
			for (sub_dir_path, sub_dir, file_names) in walk(subpath): 
				print('lista sub_dir_names = ',sub_dir)
				exam_name = sub_dir
				break


			destinoantes = destinofinal # pra evitar acumular
			for j in range(0, len(exam_name)): # ENTRA nas pastas de Exames 

				#print('exam_name = ',exam_name[j])
				exam_path = subpath+'/'+exam_name[j]
				nome_do_exame = exam_name[j]
				print('exam_path = ',exam_path)

				# cria o novo dir
				destinofinal = destinoantes # pra evitar acumular
				destinofinal = destinofinal+'/'+exam_name[j] # termina de construir o destino
				if not os.path.exists(destinofinal):
					os.makedirs(destinofinal) 


				# Pegar a máscara Mask.jpg
				import matplotlib.image as img
				mask = img.imread(exam_path+'/Mask.jpg')
				#mat_conv = img.imread(exam_path+'/Mask.jpg')
				#print (mask)



				for (paths, dirs, arq_names) in walk(exam_path+'/'+'Registradas'): # ENTRA na pasta Registradas
					#print('lista arquivos dentro de registradas = ',arq_names)
					#print (arq_names)

					for k in range(0, len(arq_names)):

						if arq_names[k].find(".txt") != -1:

							# Lê as matrizes de temperaturas
							#print ('\nachou um txt  = ', arq_names[k])
							import numpy as np
							matrix = np.loadtxt(exam_path + '/' + 'Registradas' +'/'+ arq_names[k])

							mat_conv = np.zeros((matrix.shape[0], matrix.shape[1])) # matriz



							#print (mask.shape[0]) # 480 , 640
							for x in range(0, mask.shape[0]):
								for y in range(0, mask.shape[1]):
									if(mask[x][y] < 128 ):	# limiarizar os ruídos
										#mask[x][y] = 0 		# limpar os ruidos do fundo
										matrix[x][y] = 0 	# zerar o fundo da matriz de temperatura
									#else:
										#mask[x][y] = 255	# limpar os ruidos da máscara							

									
									if(maior < matrix[x][y]):
										maior = matrix[x][y] 

									if( (menor > matrix[x][y]) and (matrix[x][y] != 0) ):
										menor = matrix[x][y]  


									# Conversao de Temperatura para pixel
									t = matrix[x][y]
									cinzamax = 255
									cinzamin = 0

									tmax = 36   # pois o maior valor (35.922) produz o maior valor de pixel 253, que nao pode passar o 255
									tmin = 22.4 # pois o menor valor (22.56) produz o menor valor de pixel 3, que nao pode ser 0

									# outras temps 35.67, 26.27 | 
									# maior e menor universal 35.922, 22.56
									# 36, 26

									#exam_path =  D:\DMR\BASE-lincoln/saudavel/ID_172
									#maior, menor =  34.941 ,  24.8 # terceiro menor
									#exam_path =  D:\DMR\BASE-lincoln/saudavel/ID_226
									#maior, menor =  34.941 ,  23.12 # segundo menor
									#exam_path =  D:\DMR\BASE-lincoln/saudavel/ID_220
									#maior, menor =  34.941 ,  22.56 # primeiro menor								


									# conversao
									p = int(round( cinzamax - ((tmax - t)/(tmax- tmin))*(cinzamax-cinzamin) ))

									#p = 50			
									if (p<0):
										p = 0									
										#print(int(p), t)

									mat_conv[x][y] = int(p) # atribuicao	



							''' # APENAS TESTE ### PARTE PARA PEGAR O MAIOR E MENOR DE CADA IMAGEM
							mai = 0
							men = 1000
							for x in range(0, mask.shape[0]):
								for y in range(0, mask.shape[1]):
									#if(mat_conv[x][y] > 0 ):	
									#	print (mat_conv[x][y])
									
									if(mai < mat_conv[x][y]):
										mai = mat_conv[x][y] 

									if( (men > mat_conv[x][y]) and (mat_conv[x][y] != 0) ):
										men = mat_conv[x][y]  		
							'''

							#print ('mai, men = ', mai, ', ', men) # pixel
							#print ('maior, menor = ', maior, ', ', menor) # temp

							#print('matrix = \n',  matrix)

							#mat_conv[200][200] = 255
							#print (mat_conv[200][200])

							nome_arq_salvar = destinofinal+'/'+exam_name[j] # sem o numero e a extensao
							#print (arq_names[k])

							# SE o arquivo termina no num 2.txt, nomear, como 02.txt - Ex.
							sufixo = ''
							if ((arq_names[k].find("_0.txt") != -1) or (arq_names[k].find("-0.txt") != -1) ):
								sufixo = '-00'							
							if ((arq_names[k].find("_1.txt") != -1) or (arq_names[k].find("-1.txt") != -1) ):
								sufixo = '-01'
							if ((arq_names[k].find("_2.txt") != -1) or (arq_names[k].find("-2.txt") != -1) ):
								sufixo = '-02'
							if ((arq_names[k].find("_3.txt") != -1) or (arq_names[k].find("-3.txt") != -1) ):
								sufixo = '-03'
							if ((arq_names[k].find("_4.txt") != -1) or (arq_names[k].find("-4.txt") != -1) ):
								sufixo = '-04'
							if ((arq_names[k].find("_5.txt") != -1) or (arq_names[k].find("-5.txt") != -1) ):
								sufixo = '-05'
							if ((arq_names[k].find("_6.txt") != -1) or (arq_names[k].find("-6.txt") != -1) ):
								sufixo = '-06'
							if ((arq_names[k].find("_7.txt") != -1) or (arq_names[k].find("-7.txt") != -1) ):
								sufixo = '-07'
							if ((arq_names[k].find("_8.txt") != -1) or (arq_names[k].find("-8.txt") != -1) ):
								sufixo = '-08'
							if ((arq_names[k].find("_9.txt") != -1) or (arq_names[k].find("-9.txt") != -1) ):
								sufixo = '-09'
							if ((arq_names[k].find("_10.txt") != -1) or (arq_names[k].find("-10.txt") != -1) ):
								sufixo = '-10'
							if ((arq_names[k].find("_11.txt") != -1) or (arq_names[k].find("-11.txt") != -1) ):
								sufixo = '-11'
							if ((arq_names[k].find("_12.txt") != -1) or (arq_names[k].find("-12.txt") != -1) ):
								sufixo = '-12'
							if ((arq_names[k].find("_13.txt") != -1) or (arq_names[k].find("-13.txt") != -1) ):
								sufixo = '-13'
							if ((arq_names[k].find("_14.txt") != -1) or (arq_names[k].find("-14.txt") != -1) ):
								sufixo = '-14'
							if ((arq_names[k].find("_15.txt") != -1) or (arq_names[k].find("-15.txt") != -1) ):
								sufixo = '-15'
							if ((arq_names[k].find("_16.txt") != -1) or (arq_names[k].find("-16.txt") != -1) ):
								sufixo = '-16'
							if ((arq_names[k].find("_17.txt") != -1) or (arq_names[k].find("-17.txt") != -1) ):
								sufixo = '-17'
							if ((arq_names[k].find("_18.txt") != -1) or (arq_names[k].find("-18.txt") != -1) ):
								sufixo = '-18'
							if ((arq_names[k].find("_19.txt") != -1) or (arq_names[k].find("-19.txt") != -1) ):
								sufixo = '-19'
							if ((arq_names[k].find("_20.txt") != -1) or (arq_names[k].find("-20.txt") != -1) ):
								sufixo = '-20'

							np.savetxt(nome_arq_salvar+sufixo+'.txt', matrix, delimiter=' ', fmt='%1.3f')
							import cv2
							cv2.imwrite(nome_arq_salvar+sufixo+'.png', mat_conv)
	
					
					print ('maior, menor = ', maior, ', ', menor )

					#return
					break



		print ('maior, menor = ', maior, ', ', menor )

		print ('\nFIM\n')
		return # para só entrar nos subs SAUDAVEL e DOENTE

		print('Done ', len(dirr))
   
	print('Done Dataset !\n')





base_origem = 'D:\DMR\BASE-lincoln'
#base_origem = 'D:\DMR\_B'
#base_origem = 'D:\DMR\Baseteste2'
destino = 'D:\DMR\BASE-lincoln-ROI-PNG'


load_dataset(base_origem, destino)
