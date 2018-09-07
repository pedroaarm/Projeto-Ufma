import csv

#classe para formato e salvar os dados no arquivo .csv

class FormatData:
	def __init__(self,name='',classe='',features=[]):
		self.classe=classe
		self.name=name
		self.features=features
		
	def __str__(self):
		#formata no padrão dos arquivos .csvfile	
		OutString="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46},{47},{48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60}".format(self.classe,self.features[0],self.features[1],self.features[2],self.features[3],self.features[4],self.features[5],self.features[6],self.features[7],self.features[8],self.features[9],self.features[10],self.features[11],self.features[12],self.features[13],self.features[14],self.features[15],self.features[16],self.features[17],self.features[18],self.features[19],self.features[20],self.features[21],self.features[22],self.features[23],self.features[24],self.features[25],self.features[26],self.features[27],self.features[28],self.features[29],self.features[30],self.features[31],self.features[32],self.features[33],self.features[34],self.features[35],self.features[36],self.features[37],self.features[38],self.features[39],self.features[40],self.features[41],self.features[42],self.features[43],self.features[44],self.features[45],self.features[46],self.features[47],self.features[48],self.features[49],self.features[50],self.features[51],self.features[52],self.features[53],self.features[54],self.features[55],self.features[56],self.features[57],self.features[58],self.features[59]) 
		return OutString

	def saveData(fileName='',Data=[]):
		#salva os dados em um arquivo .csv
		with open(fileName,"a",newline='') as csvfile:
			DataWriter = csv.writer(csvfile,delimiter='\n',quotechar=' ',quoting=csv.QUOTE_NONNUMERIC)
			DataWriter.writerow(Data)
			csvfile.close()




