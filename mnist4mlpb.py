#Este programa prueba conjuntamente (para fines comparativos) una red neuronal de 3 capas 
#y un regresor logistico pre-entrenados para identificacion de digitos MNIST. 
#Se cargan imagenes al azar de MNIST_TEST, se despliegan visualmente y el regresor logistico trata de identificar la imagen
import numpy as np
import pickle
import matplotlib.pyplot as plt
        
data = np.loadtxt('mnist_test.csv', delimiter=',') 
ncol = data.shape[1]
# definiendo entradas y salidas
X_test = data[:,1:ncol]
y_test = data[:,0]

# se genera una matriz de 28x28 para guardar imagen en escala de grises
select_image = np.random.random([28, 28])

# se escoge al azar un indice de imagen de prueba
ex = np.random.randint(0, 10000)

# se carga el regresor logistico entrenado
loaded_model1 = pickle.load(open('finalized_model.sav', 'rb'))

# se carga la red neuronal entrenada
loaded_model2 = pickle.load(open('finalized_model2.sav', 'rb'))

# el regresor logistico trata de identificar la imagen
xtest = X_test[ex,].reshape(1, -1)
predicted = loaded_model1.predict(xtest)
print("La regresion logistica predice un:")	
print(predicted)

# la red neuronal trata de identificar la imagen
xtest = X_test[ex,].reshape(1, -1)
predicted = loaded_model2.predict(xtest)
print("La red neuronal predice un:")	
print(predicted)

print("El numero realmente es un:")	
print(y_test[ex])

# se traduce el vector seleccionado a imagen y se despliega visualmente
select_image1 = 1 - X_test[ex,] / 255
k = 0
for i in range(0,28):
    for j in range(0,28):
        select_image[i,j] = select_image1[k] 
        k = k + 1      
   
plt.imshow(select_image, cmap='gray', interpolation='nearest')
plt.show()