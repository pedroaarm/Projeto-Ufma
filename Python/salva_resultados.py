import csv

class FormatData:
	def __init__(self,c_parameter,result,scaler,faixa='',kernel='',test_valida=''):
		self.faixa=faixa
		self.c_parameter=c_parameter
		self.kernel=kernel
		self.result=result
		self.scaler=scaler
		self.test_valida=test_valida
		
	def __str__(self):
		#formata no padrão dos arquivos .csvfile	
		OutString="{0};{1};{2};{3};{4};{5}".format(self.faixa,self.kernel,self.c_parameter,self.scaler,self.result,self.test_valida) 
		return OutString

	def saveData(fileName='',Data=[]):
		#salva os dados em um arquivo .csv
		with open(fileName,"a",newline='') as csvfile:
			DataWriter = csv.writer(csvfile,delimiter='\n',quotechar=' ',quoting=csv.QUOTE_NONNUMERIC)
			DataWriter.writerow(Data)
			csvfile.close()
