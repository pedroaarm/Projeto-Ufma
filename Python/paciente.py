import images
#classe paciente que guarda o nome do exame, a classe patologica, uma lista com as imagens
#e um vetor com as caracteristicas

class Patient:
	def __init__ (self,Name='',classe='',images=[],features=[]):
		
		self.Name=Name
		self.classe=classe
		self.images=images
		self.features=features
	'''
	def __str__(self):
		OutString="'{0}', {1}, {2}".format(self.Name,self.images,self.features)
		return OutString
	'''
	def GetName(self):
		return self.Name

	def SetName(self,Name):
		self.Name=Name

	def GetClasse(self):
		return self.classe

	def SetClasse(self,classe):
		self.classe=classe		
		
	def GetImages(self):
		return self.images

	def SetImages(self,images):
		self.images=images

	def GetFeatures(self):
		return self.features

	def SetFeatures(self,vector):
		self.features=vector

	def AddFeature(self,info):
		self.features.append(info)	

	def AddImage(self,image):
		self.images.append(image)
