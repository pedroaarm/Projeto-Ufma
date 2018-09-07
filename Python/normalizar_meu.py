
def normalize_graph(G): # retorna o menor e menor peso de aresta do grafo  
	menor = 1000
	maior = 0
	edges = G.edges()

	# primeiro passeio para saber o maior e menor
	for u,v in edges:
		peso = G[int(u)][int(v)]['w']
		if peso > maior: # se o peso for maior que o maior
			maior = peso
		if peso < menor: # se o peso for menor que o menor
			menor = peso
	if maior == menor:
		return G		

	# segundo passeio para aplicar a normalização (0,1)
	for u,v in edges:
		peso = G[int(u)][int(v)]['w']
		#G[int(u)][int(v)]['w'] = (peso - menor)/(maior - menor)
		G[int(u)][int(v)]['w'] =  1 - ((maior - peso)/(maior- menor))*(1-0.01) # 0.01 é o limite inferior
	
	return G 	
