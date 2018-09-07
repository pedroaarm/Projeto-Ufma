# ESSE AUGMENTATION CONSIDERA CADA SUBPASTA DE EXAME DE ITD 
# ALTERANDO O CONTRASTE 
# PODE ESCOLHER SE RESIZE E/OU AUG 
# Augmentation é usado para aumentar a base de treinameto para Deep Learning



# Imports
from PIL import Image
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from os import walk
import numpy as np
import os


from PIL import Image
import random

qtd_aug = 16

# funcao de contraste simples. Nao conhecida formula
def change_contrast(img, level):
	factor = (259 * (level + 255)) / (255 * (259 - level))
	def contrast(c):
		return 128 + factor * (c - 128)

	return img.point(contrast)


def bounding_box(im):
	x1 = im.getbbox()[0] - 30
	y1 = im.getbbox()[1] - 20
	x2 = im.getbbox()[2] + 30
	y2 = im.getbbox()[3] + 30

	tamx = 639 # dimensoes da imagem original
	tamy = 479

	if(x1<0):
		x1=0
	if(y1<0):
		y1=0
	if(x2>tamx):
		x2=tamx
	if(y2>tamy):
		y2=tamy

	area = (x1, y1, x2, y2)
	cropped_img = im.crop(area)
	return cropped_img




# Data Augmentation
data_gen = ImageDataGenerator(# Amount of rotation
							  rotation_range=10,
	
							  # Amount of shift
							  width_shift_range=0.02,
							  height_shift_range=0.1,
	
							  # Shear angle in counter-clockwise direction as radians
							  shear_range=0.06,
	
							  # Range for random zoom
							  zoom_range= [1, 1+0.2],
	
							  # Boolean (True or False). Randomly flip inputs horizontally
							  horizontal_flip=True,
	
							  # Points outside the boundaries of the input are filled
							  # according to the given mode
							  fill_mode='nearest')

# Import folder
def load_dataset(path, dest, size, fazer_aug, random_contrast, bb):
	print ("Gerando imagens em:")
	print (dest)
	print("\n")

	f = [] # folder    
	dirr = [] # dir from this folder 
	# Find all subfolders in folder
	for (dir_path, dir_names, file_names) in walk(path):
		f.extend(dir_names)
		dirr.extend(dir_names)
		#print('fff = ',f)

		i = 0
		for i in range(0, len(dir_names)):
			#print('sfff = ',f)

			exame = path+'/'+f[i]
			#print('exame = ', exame)

			
			#for sub in exame:
			#print ('subfolder = ', sub)
			files = []
			for (dir_path, dir_names, file_names) in walk(exame):
				files.extend(file_names)


				# For each image in folder
				for item in files:
					if not os.path.isdir(dest): # cria o diretorio
						os.makedirs(dest)

					#print ('dir_img = ', exame+'/'+item)
					image = Image.open(exame+'/'+item).convert('L')

					if(bb==1):
						image = bounding_box(image)

					# aplica contraste
					if (random_contrast==1):
						grid = random.randint(8, 16)
					else:
						grid = 12
					# convert pil to opencv
					img = np.array(image) 
					# create a CLAHE object (Arguments are optional).
					clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(grid,grid))
					cl1 = clahe.apply(img)
					cv2.imwrite('exit.jpg',cl1)
					# convert opencv to pil
					image = Image.fromarray(cl1)					

					#image.thumbnail(size, Image.ANTIALIAS)
					#image.thumbnail(size, Image.ANTIALIAS) ## redimensionar sem alterar as dimensões   
					image = image.resize(size, Image.ANTIALIAS) ## redimensionar alterando as dimensões    


					if(fazer_aug==0): # se nao for aug
						image.save(dest+'/'+dirr[i]+'_'+item, "JPEG", quality=100, optimize=True, progressive=True)

					if(fazer_aug==1): # se for aug
						# Create a numpy array with shape (1, 500, 500)
						x = img_to_array(image)
						#x = np.asarray(x)
						
						# Convert to a numpy array with shape (1, 1, 500, 500)
						x = x.reshape((1,) + x.shape)
						
						i = 0
						for batch in data_gen.flow(x, save_to_dir=dest, save_prefix='DA', save_format='jpg', batch_size=1000, shuffle=True):
							i += 1
							if i > qtd_aug: # 16
								break						
				
		print('Done ', str(i), '/', len(dirr))
   
	print('Done Dataset !\n')



size = (100,75)
base_origem = 'D:/DMR/BaseCNN/BASE_cnn/bases_k-folds_5-70_ITE/fold'


for i in range(1, 6): # 1 a 5
	dir_base = base_origem + str(i) +'/'
	destino = base_origem + str(i)+ '_aug_2/' # ele cria o diretorio 
	
	sub = 'teste/'
	load_dataset(dir_base+sub+'A', destino + sub+'A', size, 0, 0, 0)
	load_dataset(dir_base+sub+'B', destino + sub+'B', size, 0, 0, 0)

	sub = 'treino/'
	load_dataset(dir_base+sub+'A', destino + sub+'A', size, 1, 1, 0)
	load_dataset(dir_base+sub+'B', destino + sub+'B', size, 1, 1, 0)
	
	qtd_aug = 4
	sub = 'val/'
	load_dataset(dir_base+sub+'A', destino + sub+'A', size, 1, 1, 0)
	load_dataset(dir_base+sub+'B', destino + sub+'B', size, 1, 1, 0)

	sub = 'val/'
	load_dataset(dir_base+sub+'A', destino + sub+'A', size, 0, 1, 0)
	load_dataset(dir_base+sub+'B', destino + sub+'B', size, 0, 1, 0)
	