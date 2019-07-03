import pandas
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('Energy_efficiency.csv')

X_entrenamiento = data.iloc[:,0:7]
y_entrenamiento1 = data.iloc[:,8]
y_entrenamiento2 = data.iloc[:,9]

linealregression1 = LinearRegression()
linealregression2 = LinearRegression()

linealregression1.fit(X_entrenamiento, y_entrenamiento1)
linealregression2.fit(X_entrenamiento, y_entrenamiento2)

linealregression1.score(X_entrenamiento, y_entrenamiento1)
linealregression2.score(X_entrenamiento, y_entrenamiento2)
