import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
print(np.shape(imagenes), n_imagenes) # Hay 1797 digitos representados en imagenes 8x8
# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
# Vamos a hacer un split training test
scaler = StandardScaler()
#StandardScaler resta la media y pone varianza cero.
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# Vamos a entrenar solamente con los digitos iguales a 1
numero = 1 #¿Cómo sabe que los datos son uno? Están etiquetados en el atributo target.
dd = y_train==numero
#matriz de covarianza de pca
cov = np.cov(x_train[dd].T)
#valores y vectores propios
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
#hay que elegir los valores más importantes (grandes)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

componentes_pca=np.arange(4,40)
F1_test=[]
F1_train=[]
for features in componentes_pca:
    x_train_pca=np.dot(x_train[:features,:],vectores[:,:features])
    x_test_pca=np.dot(x_test[:features,:],vectores[:,:features])
    unos=np.mean(x_train[dd],axis=0)

    LDA=LinearDiscriminantAnalysis()
    LDA.fit(X=x_train_pca,y=y_train[:features])
    y_train_LDA=(LDA.predict(x_train_pca))
    y_test_LDA=(LDA.predict(x_test_pca))
    F1_train.append(f1_score(y_train[:features],y_train_LDA,average='micro'))
    F1_test.append(f1_score(y_test[:features],y_train_LDA,average='micro'))
plt.scatter(componentes_pca,F1_train,label='train (50%)')
plt.scatter(componentes_pca,F1_test,label='test(50%)')
plt.legend()
plt.title('Clasificación UNO')
plt.xlabel('Componentes de PCA')
plt.ylabel('F1')
plt.savefig('F1_score_LDA.png')
