
from PIL import Image

dirr = 'D:/DMR/BaseCNN/BASE_cnn/bases_k-folds_5-70_ITE/fold1_aug_2/teste/B/ID_9_'

for i in range(1, 21): 
	nome = dirr
	nome = nome+str(i)+'.jpg'

	image = Image.open(nome).convert('L')

	area = (30, 30, 540, 380)
	
	cropped_img = image.crop(area)
	cropped_img.save(nome, "JPEG", quality=100, optimize=True, progressive=True)