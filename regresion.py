# Este programa entrena una regresion logistica para MNIST y guarda el modelo entrenado para posterior uso.
# importando sklearn y numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

data = np.loadtxt('mnist_train.csv', delimiter=',') 
print("Lectura de la base de datos completa")
ncol = data.shape[1]
# definiendo entradas y salidas
X = data[:,1:ncol]
y = data[:,0]

clf = LogisticRegression(
    solver='lbfgs', 
    max_iter=300, 
    verbose=1, 
    random_state=1
)
clf.fit(X, y)

print("Entrenamiento completo")

data = np.loadtxt('mnist_test.csv', delimiter=',') 

ncol = data.shape[1]

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
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
