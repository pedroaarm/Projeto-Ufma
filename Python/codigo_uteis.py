

'''
from PIL import Image
im = Image.open('exit.jpg')
im_grey = im.convert('LA') # convert to grayscale
width,height = im.size

total=0
for i in range(0,width):
    for j in range(0,height):
        total += im_grey.getpixel((i,j))[0]

mean = total / (width * height)
print (mean)


imagemm = cv2.imread('exit.jpg', cv2.IMREAD_GRAYSCALE)
print (imagemm[200][200][0])
'''

image3 = io.imread('exit.jpg') # lê com todos os canais
image1 = io.imread('mama.jpg') # lê com todos os canais

imageA1 = image3[:, :, 0]  # deixa só um canal
print (image1.shape)
print (imageA1.shape)

print (np.dstack((image1, imageA1)).shape) # juntar matriz na 3ª dimensao


#image_orig = image_orig[:, :, 0]  # deixa só um canal

#image_orig = Image.open('exit.jpg').convert('L')


#print (image.shape[0]) # 480
#print (image.shape[1]) # 640
#(h, w) = image.shape[:2]

	






# LER E SALVAR imagens
#img.imsave(nome_arq_salvar+sufixo+'.png', mat_conv)
#scipy.misc.toimage(mat_conv).save(nome_arq_salvar+sufixo+'.png')
#scipy.misc.imsave(nome_arq_salvar+sufixo+'.png', mat_conv) # adicionar imagem com temp convertida

####misc.toimage(img, cmin=0, cmax=100)
#import cv2
#cv2.imwrite("filename.png", np.zeros((10,10)))
#cv2.imwrite(nome_arq_salvar+sufixo+'.png', mat_conv)

#import numpy as np
#import cv2
#nArray = np.zeros((480, 640), np.float32) #Create the arbitrary input img
#cArray2 = nArray
#cv2.imwrite("cpImage.bmp", cArray

#from PIL import Image
i#mg = Image.fromarray(mat_conv)
#if img.mode != 'RGB':
#	img = img.convert('RGB')
#img.save(nome_arq_salvar+sufixo+'.png')
