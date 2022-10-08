from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import scikitplot as skplt
import matplotlib.pyplot as plt

#carrega a base de dados
iris = load_iris()
entradas = iris.data
alvos = iris.target
nomes_das_classes = iris.target_names

#crias as bases de treinamento e teste
Ent_tre, Ent_test, Alvo_tre, Alvo_test = train_test_split(entradas,alvos,test_size=0.3, random_state=10, stratify= alvos)

# cria o modelo neural ( com uma unica camada de 5 neuronios)
net = MLPClassifier(solver='lbfgs', max_iter=500, hidden_layer_sizes=(5),verbose=True)

# realiza o treinamento do modelo neural
modelo_ajustado = net.fit(Ent_tre,Alvo_tre)

# estima a presisão do modelo treinado
score = modelo_ajustado.score(Ent_test, Alvo_test)
print(score)

# calcula as previsoes do modelo
previsoes = modelo_ajustado.predict(Ent_test)
prevpb = modelo_ajustado.predict_proba(Ent_test)


precisao = accuracy_score(Alvo_test, previsoes)
print(precisao)

print(classification_report(Alvo_test, previsoes))

confusao = confusion_matrix(Alvo_test, previsoes)
print(confusao)

opcoes_titulos = [('Matriz de confusão sem normalização',None),('Matriz de confusão normalizada','true')]
for titulo, norm in opcoes_titulos:
    disp = plot_confusion_matrix(modelo_ajustado, Ent_test, Alvo_test, display_labels=nomes_das_classes,cmap=plt.cm.Blues,normalize= norm)
    disp.ax_.set_title(titulo)

    print(titulo)
    print(disp.confusion_matrix)

plt.show()

#plot usando a biblioteca scikitplot
skplt.metrics.plot_confusion_matrix(Alvo_test, previsoes)
plt.show()
skplt.metrics.plot_confusion_matrix(Alvo_test, previsoes, normalize='True')
plt.show()

# plot a ROC
skplt.metrics.plot_roc(Alvo_test,prevpb)
plt.show()

skplt.metrics.plot_precision_recall(Alvo_test, prevpb)
plt.show()