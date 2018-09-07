# Este programa aplica CLAHE e depois SLIC para obter a máscara de superpixels

# ENTRADA: imagem individual, cinza ou colorida
# SAIDA: imagem CLAHE e varias mascaras SLIC com determinada qtd de superpixels. Tudo cinza


# CLAHE 
def clahe(nome_imagem):
	import numpy as np
	import cv2
	import numpy as np
	import os
	import random
	from PIL import Image

	grid = random.randint(16, 16)
	imagem = Image.open(nome_imagem).convert('L')
	# convert pil to opencv
	img = np.array(imagem) 

	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(grid,grid))
	cl1 = clahe.apply(img)
	cl2 = cv2.medianBlur(cl1, 5)

	cv2.imwrite('exit_clahe_16.png',cl1)
	cv2.imwrite('exit_clahe_median_5.png',cl2)

	# convert opencv to pil
	img = Image.fromarray(cl1)

	return cl2
	# FIM clahe()




def quantizacao(entrada):
	from sklearn.cluster import MiniBatchKMeans
	import numpy as np
	import argparse
	import cv2

	# load the image and grab its width and height
	# image = cv2.imread('exit_clahe_16_med_5.jpg') # exit_clahe_16_med_5.jpg' # Entrada
	image = entrada
	(h, w) = image.shape[:2]

	# convert the image from the RGB color space to the L*a*b*
	# color space -- since we will be clustering using k-means
	# which is based on the euclidean distance, we'll use the
	# L*a*b* color space where the euclidean distance implies
	# perceptual meaning

	image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # se a entrada for colorida

	#image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # se a entrada for gray
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # se a entrada for gray

	# reshape the image into a feature vector so that k-means
	# can be applied
	image = image.reshape((image.shape[0] * image.shape[1], 3))

	# apply k-means using the specified number of clusters and
	# then create the quantized image based on the predictions
	clt = MiniBatchKMeans(n_clusters = 8)
	labels = clt.fit_predict(image)
	quant = clt.cluster_centers_.astype("uint8")[labels]

	# reshape the feature vectors to images
	quant = quant.reshape((h, w, 3))
	image = image.reshape((h, w, 3))

	# convert from L*a*b* to RGB
	quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
	image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

	cv2.imwrite('exit_clahe_16_med_5_quant_8.png',quant)

	# display the images and wait for a keypress
	#cv2.imshow("image", np.hstack([image, quant]))
	
	return quant








def slic(cl2):	

	# SLIC
	from skimage.segmentation import slic
	from skimage.segmentation import mark_boundaries
	from skimage.util import img_as_float
	from skimage import io
	import matplotlib.pyplot as plt
	import argparse
	import numpy as np
	import cv2
	from PIL import Image
	import scipy.misc # salvar mat float como img


	# load the image and convert it to a floating point data type

	image_orig = cl2 
	# Entrada
	#image_orig = io.imread('mama_clahe_gray.jpg') # lê com todos os canais

	#image_orig = image_orig[:, :, 0]  # deixa só um canal 
	#### comentar, se tiver 1 canal. ou Fazer if

	image = img_as_float(image_orig) # converte em float (0 a 1)
	canal = img_as_float(image_orig) 

	# image tem canal triplicado, para passar pelo SLIC 
	image = np.dstack((image, canal)) # juntar matriz na 3ª dimensao, 2
	image = np.dstack((image, canal)) # juntar matriz na 3ª dimensao, 3

	# loop over the number of segments
	for numSegments in (100, 200, 400, 600, 800):

		for y in range(0, image.shape[0]): # 480 y
			for x in range(0, image.shape[1]): # 640 x
				if(image[y][x][0]<0.1176): # 0.1176 é conversao do limiar 30 (0 a 255) para float (0 a 1)
					image[y][x][0] = 0 # tira os ruidos do fundo
					image[y][x][1] = 0
					image[y][x][2] = 0
					image_orig[y][x] = 0 # tira os ruidos do fundo

		# apply SLIC and extract (approximately) the supplied number of segments
		segments = slic(image, n_segments = numSegments, sigma = 1, multichannel=True, enforce_connectivity=True, max_iter=10)
		#print (segments[300][200])
		#image, n_segments=100, compactness=10.0, max_iter=10, sigma=0, spacing=None, multichannel=False, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False)


		for y in range(0, image.shape[0]): # 480 y
			for x in range(0, image.shape[1]): # 640 x
				if(image[y][x][0]<0.1176): # 0.1176 é comversao do limiar 30 (0 a 255) para float (0 a 1)
					segments[y][x] = -1 # desconsidera os superpixels do fundo
					image[y][x][0] = 0


		# show the output of SLIC
		fig = plt.figure("slic-%d-exit_clahe_16_med_11" % (numSegments))
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(mark_boundaries(image, segments))
		plt.axis("off")

		image_marcacoes = mark_boundaries(image, segments)
		scipy.misc.imsave('image_marcacoes'+str(numSegments)+'.png', image_marcacoes)

		scipy.misc.imsave('image_segments'+str(numSegments)+'.png', segments)	

		#scipy.misc.toimage(image, cmin=0.0, cmax=255)
		#scipy.misc.imsave('image'+str(numSegments)+'.png', image)

		scipy.misc.imsave('image_orig.png', image_orig)



		#plt.savefig('foo_'+str(numSegments)+'.png') # caso queira salvar a plotagem

	# show the plots
	#plt.show()

	return segments
	# FIM slic()






# MAIN

nome_imagem = 'mama.jpg'
img_clahe = clahe(nome_imagem)
# tt = quantizacao()
segments = slic(img_clahe)
print('feito')








'''
# QUANTIZACAO ORIGINAL
# import the necessary packages
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2

# load the image and grab its width and height
image = cv2.imread('exit_clahe_16_med_5.jpg') # exit_clahe_16_med_5.jpg'
(h, w) = image.shape[:2]

# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning

image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # se a entrada for colorida

#image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # se a entrada for gray
#image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # se a entrada for gray

# reshape the image into a feature vector so that k-means
# can be applied
image = image.reshape((image.shape[0] * image.shape[1], 3))

# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
clt = MiniBatchKMeans(n_clusters = 8)
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]

# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))

# convert from L*a*b* to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

cv2.imwrite('exit_clahe_16_med_5_quant_8.jpg',quant)

# display the images and wait for a keypress
#cv2.imshow("image", np.hstack([image, quant]))
cv2.waitKey(0)
'''


'''
https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/
https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
'''