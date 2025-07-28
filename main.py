import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

datos = pd.read_csv('housing.csv')
datos.dropna(inplace=True)
datos = pd.get_dummies(datos, columns=['ocean_proximity'], drop_first=True)

caracteristicas = datos.drop('median_house_value', axis=1)
etiqueta = datos['median_house_value']

carac_entrenamiento, carac_prueba, etiqueta_entrenamiento, etiqueta_prueba = train_test_split(
    caracteristicas, etiqueta, test_size=0.2, random_state=42
)

escalador = StandardScaler()
carac_entrenamiento_escalado = escalador.fit_transform(carac_entrenamiento)
carac_prueba_escalado = escalador.transform(carac_prueba)

modelo_arbol = DecisionTreeRegressor(max_depth=5, random_state=42)
modelo_arbol.fit(carac_entrenamiento, etiqueta_entrenamiento)
predicciones_arbol = modelo_arbol.predict(carac_prueba)

r2_arbol = r2_score(etiqueta_prueba, predicciones_arbol)
rmse_arbol = np.sqrt(mean_squared_error(etiqueta_prueba, predicciones_arbol))

print("🔎 Arbol de Decision:")
print(f"  R^2: {r2_arbol:.4f}")
print(f"  RMSE: {rmse_arbol:.2f}")

plt.figure(figsize=(20, 10))
plot_tree(modelo_arbol, feature_names=caracteristicas.columns, filled=True)
plt.title("Arbol de Decision para Prediccion de Valor de Viviendas")
plt.savefig("arbol_decision.png")
plt.close()

modelo_svm = SVR(kernel='rbf', C=100, epsilon=500, gamma=0.1)
modelo_svm.fit(carac_entrenamiento_escalado, etiqueta_entrenamiento)
predicciones_svm = modelo_svm.predict(carac_prueba_escalado)

r2_svm = r2_score(etiqueta_prueba, predicciones_svm)
rmse_svm = np.sqrt(mean_squared_error(etiqueta_prueba, predicciones_svm))

print("\n🤖 Maquina de Vectores de Soporte (SVM) [ajustada]:")
print(f"  R^2: {r2_svm:.4f}")
print(f"  RMSE: {rmse_svm:.2f}")

plt.figure(figsize=(12, 8))
sns.heatmap(datos.corr(), annot=True, cmap='YlGnBu')
plt.title("Mapa de Correlacion entre Variables")
plt.savefig("mapa_correlacion.png")
plt.close()

variables_relacionadas = ['median_income', 'housing_median_age', 'total_rooms', 'median_house_value']
sns.pairplot(datos[variables_relacionadas])
plt.suptitle("Relaciones entre Variables Relevantes", y=1.02)
plt.savefig("relaciones_variables.png")
plt.close()
