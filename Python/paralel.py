# https://social.stoa.usp.br/igordsm/blog/processamento-paralelo-usando-python-parte-1

# TRATA-SE DE UM TESTE PARA EXECUCAO EM SERIAL E EM PARALELO
# RESULTADOS no notebook de Stefano (com 4 nucleos, funcao com 10 vezes, lista tamanho 1000, matriz 400x400):
# Versao Paralela: 46.52865524563654
# Versao Serial: 63.150846421889376

import numpy as np
import time
import multiprocessing as mp

num_vezes = 2
tamlista = 1000
dimensao = 400

def funcao_demorada(el):
	# isto não faz sentido algum...
	A, B = el
	A += B
	A *= B
	A = np.dot(A, B)
	A = np.linalg.inv(A)
	A = np.dot(B, A)
	B = np.dot(A, B)
	A = np.linalg.inv(A)
	A = np.dot(A, B)
	B = np.dot(B, A)
	A = np.dot(A, B)
	return A

def cria_lista(n, sz=(dimensao, dimensao)):
	return [(np.random.rand(sz[0], sz[1]), np.random.rand(sz[0], sz[1])) for i in range(n)]

def versao_serial():
	#lista = cria_lista(tamlista)
	resultados = []
	for el in lista:
		resultados.append(funcao_demorada(el))
	#print ('\n', len(resultados),'x', len(resultados[0]),'\n', resultados, '\n\n')

def versao_paralela():
	#print ('mp.cpu_count() = ', mp.cpu_count()) # numero de nucleos de processamento
	p = mp.Pool(mp.cpu_count())
	#lista = cria_lista(tamlista)
	resultados = p.map(funcao_demorada, lista)
	#print ('\n', len(resultados),'x', len(resultados[0]),'\n', resultados, '\n\n')

def executa_varias_vezes(func, n):
	tempos = []
	for i in range(n):
		start = time.clock()
		func()
		tempos.append(time.clock() - start)
	return tempos

lista = cria_lista(tamlista) # global nesse caso, mas pode não ser

if __name__ == "__main__":
	print('\nVersao Paralela')
	tparalela = executa_varias_vezes(versao_paralela, num_vezes)
	print('Paralela:', sum(tparalela)/num_vezes)

	print('\nVersao Serial')
	tserial = executa_varias_vezes(versao_serial, num_vezes)
	print('Serial:', sum(tserial)/num_vezes)

