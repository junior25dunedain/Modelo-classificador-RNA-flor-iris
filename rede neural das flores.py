import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def load_data():
    URL_= 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    dados = pd.read_csv(URL_,header = None)
    # dados = pd.read_csv('iris.csv',sep = ',',header = None)
    print(dados)

    dados1s = pd.DataFrame()     # outra forma de obter uma base de dados com somente os dados das flores 'Iris-setosa' e 'Iris-versicolor'
    for i in range(len(dados)):
        if dados.iloc[i, -1] == 'Iris-setosa' or dados.iloc[i, -1] == 'Iris-versicolor':
          dados1s = dados1s.append(dados.iloc[[i]], ignore_index=True)


    dados1s[4] = np.where(dados1s.iloc[:,-1]=='Iris-setosa',0,1)
    dados1s = np.asmatrix(dados1s,dtype='float64')
    return dados1s

class Perceptron(object):
    def __init__(self, num_entradas, epocas =10, taxa_aprendizagem=0.01):
        self.epocas = epocas
        self.taxa_aprendizagem = taxa_aprendizagem
        lista = []
        for n in range(num_entradas+1): # +1 é o bias
            while True:
                x = random.random()
                if x <= 0.1:
                    lista.append(x)
                    break
        self.pesos = np.array(lista)


    def cal_saida(self,entradas):
        net = np.dot(entradas,self.pesos[1:]) + self.pesos[0]
        if net >0:
            saida = 1
        else:
            saida =0
        return saida

    def treinar(self, entradas_treino, alvos):
        n_epoca = 0
        for _ in range(self.epocas):
            n_epoca = n_epoca +1
            erro = 0
            cont_erro = 0

            for entradas,alvo in zip(entradas_treino, alvos):
                estimacao = self.cal_saida(entradas)
                erro = alvo - estimacao
                if erro.all() == 0:
                    cont_erro += 1
                deltaw = self.taxa_aprendizagem*erro
                self.pesos[1:] += np.squeeze(np.asarray(deltaw*entradas))
                self.pesos[0] += deltaw
            if cont_erro == len(entradas_treino):
                break

def Dados_treino_e_teste(basedados, taxa_treino= 0.6):
    basedados = pd.DataFrame(basedados)
    entradas_treino = []
    alvos_treino = []
    alvos_teste = []
    entradas_teste = []

    cont = 1
    lista_ind = []
    while cont <= taxa_treino * len(basedados):
        j = random.randint(0, len(basedados)-1)

        if j in lista_ind:  # 70% dos dados da base de dados serão usados para o treino
            continue
        lista_ind.append(j)
        entradas_treino.append(basedados.iloc[j,:-1])
        alvos_treino.append([basedados.iloc[j,-1]])
        cont += 1

    for i in range(len(basedados)):
        if i not in lista_ind:
            entradas_teste.append(basedados.iloc[i,:-1])
            alvos_teste.append([basedados.iloc[i,-1]])

    entradas_treino = np.asmatrix(entradas_treino, dtype='float64')
    alvos_treino = np.asmatrix(alvos_treino, dtype='float64')
    entradas_teste = np.asmatrix(entradas_teste, dtype='float64')
    alvos_teste = np.asmatrix(alvos_teste, dtype='float64')

    return entradas_treino, alvos_treino, entradas_teste, alvos_teste

def Acertos(entrada_teste,alvo_teste, Perceptron):
    saida_auxiliar = []
    for i in entrada_teste:
        saida_auxiliar.append(perceptron.cal_saida(i))

    acerto = 0
    for j, k in zip(saida_auxiliar, alvo_teste):
        if j == k:
            acerto += 1

    print(f'A taxa de acerto é {round((acerto / len(alvos_teste)) * 100, 2)}%')



#main
basedados = load_data()
print(pd.DataFrame(basedados))
plt.scatter(np.array(basedados[:50,0]), np.array(basedados[:50,2]), marker = 'o', label = 'setosa')
plt.scatter(np.array(basedados[50:,0]), np.array(basedados[50:,2]), marker = 'x', label = 'versicolor')
plt.xlabel('Comprimento da pétala')
plt.ylabel('Comprimento da sépela')
plt.legend()
plt.show()

entradas_treino, alvos_treino, entradas_teste, alvos_teste = Dados_treino_e_teste(basedados,0.7)

perceptron = Perceptron(4)
perceptron.treinar(entradas_treino,alvos_treino)

Acertos(entradas_teste,alvos_teste, perceptron)

#setosa
entrada_teste = np.array([5.1,3.5,1.4,0.2])
saida_teste1 = perceptron.cal_saida(entrada_teste)
print(saida_teste1)

#versicolor
entrada_teste2 = np.array([7,3.2,4.5,1.4])
saida_teste2 = perceptron.cal_saida(entrada_teste2)
print(saida_teste2)

# virginica
entrada_teste3 = np.array([6.3,3.3,6,2.5])
saida_teste3 = perceptron.cal_saida(entrada_teste3)
print('Saida com erro --> ',saida_teste3)