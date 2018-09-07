#classe imagem que guarda o nome do arquivo, um vetor com as caracteristicas extraidas da imagem
# sua matriz de temperatura e seu histograma(normalizado)

class Image:
	def __init__(self,file='',vector=[],matrix=[],histogram=[]):
		
		self.file=file
		self.vector=vector
		self.matrix=matrix
		self.histogram=histogram
		
	def GetFile(self):
		return self.file

	def SetFile(self, file):
		self.file=file

	def GetVector(self):
		return self.vector
		
	def SetVector(self, vector):
		self.vector=vector

	def GetMatrix(self):
		return self.matrix
		
	def SetMatrix(self, matrix):
		self.matrix=matrix

	def GetHistogram(self):
		return self.histogram
		
	def SetHistogram(self, histogram):
		self.histogram=histogram
					
	def AddCharacteristic(self,info):
		self.vector.append(info)