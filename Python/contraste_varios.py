# AKI CONTEM VÁRIOS CÓDIGOS SOBRE CONTRASTE E AJUSTE DE HISTOGRAMA

'''
# Imports
from PIL import Image
import cv2
from os import walk
import numpy as np
import os

from PIL import Image
import random

# funcao de contraste
def change_contrast(img, level):
	factor = (259 * (level + 255)) / (255 * (259 - level))
	def contrast(c):
		return 128 + factor * (c - 128)

	return img.point(contrast)


image = Image.open(t).convert('L')
# muda o contraste das imagens, com range aleatorio
#nivel = random.randint(0, 10)
nivel = 10
image = change_contrast(image, nivel)
'''


# plotar histograma
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('mama.jpg',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
'''


'''
from PIL import Image
from PIL import ImageEnhance

image = Image.open('mama.jpg')
enhancer = ImageEnhance.Contrast(image)

factor = 1.4	
enhancer.enhance(factor).show("Contrast %f" % factor)
enhancer.enhance(factor).save('saida.jpg')
'''

'''
# CLAHE
import numpy as np
import cv2

img = cv2.imread('mama.jpg',0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
cl1 = clahe.apply(img)

cv2.imwrite('exit.jpg',cl1)

'''




# CLAHE 2
import numpy as np
import cv2
import numpy as np
import os
import random
from PIL import Image

grid = random.randint(16, 16)
imagem = Image.open('mama.jpg').convert('L')
# convert pil to opencv
img = np.array(imagem) 

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(grid,grid))
cl1 = clahe.apply(img)
cl2 = cv2.medianBlur(cl1, 5)

cv2.imwrite('exit_clahe_16.jpg',cl1)
cv2.imwrite('exit_clahe_median_5.jpg',cl2)
#print (cl2[200])

# convert opencv to pil
img = Image.fromarray(cl1)


