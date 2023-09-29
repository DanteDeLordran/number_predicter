# Este programa entrena una regresion logistica para MNIST y guarda el modelo entrenado para posterior uso.
# importando sklearn y numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import pickle

# leyendo la base de datos MNIST
data = np.loadtxt('mnist_train.csv', delimiter=',') 
print("Lectura de la base de datos completa")
ncol = data.shape[1]
# definiendo entradas y salidas
X = data[:,1:ncol]
y = data[:,0]

tic = time.process_time()
#clf = LogisticRegression()
clf = MLPClassifier(solver='adam', alpha=1e-5, max_iter=300, activation='logistic',
hidden_layer_sizes=(2,), verbose=True, random_state=1)
clf.fit(X, y)

toc = time.process_time()
print("Entrenamiento completo")
print("Tiempo de procesador para el entrenamiento (seg):")
print(toc - tic)   

data = np.loadtxt('mnist_test.csv', delimiter=',') 
#print(data)
ncol = data.shape[1]
# definiendo entradas y salidas
X_test = data[:,1:ncol]
y_test = data[:,0]

# predecir los valores de X_test
predicted = clf.predict(X_test)
	
# para finalizar se calcula el error
error = 1 - accuracy_score(y_test, predicted)
acurracy = accuracy_score(y_test.astype(int), predicted) * 100

print('Error : ' , error)
print('Acurracy ' , acurracy)

# el modelo entrenado se salva en disco
filename = 'finalized_model2.sav'
pickle.dump(clf, open(filename, 'wb'))
