import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos
# (Reemplaza 'ruta_datos.csv' con la ruta a tu conjunto de datos)
df = pd.read_csv('ruta_datos.csv')

# Verificar si el DataFrame se cargó correctamente
print(df.head())

# Separar características y etiqueta
# Reemplazar 'precio' con el nombre de la columna objetivo si es diferente
if 'precio' not in df.columns:
    raise ValueError("La columna objetivo 'precio' no se encuentra en el conjunto de datos. Reemplaza 'precio' con el nombre correcto de tu columna objetivo.")

X = df.drop('precio', axis=1)
y = df['precio']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo con Scikit-Learn (Regresión Lineal)
modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)
predicciones_lr = modelo_lr.predict(X_test)

# Evaluar el rendimiento del modelo
mse_lr = mean_squared_error(y_test, predicciones_lr)
r2_lr = r2_score(y_test, predicciones_lr)
print(f'Modelo Linear Regression - MSE: {mse_lr:.2f}, R2: {r2_lr:.2f}')

# Entrenar el modelo con Scikit-Learn (Arbol de Decisión)
modelo_dt = DecisionTreeRegressor()
modelo_dt.fit(X_train, y_train)
predicciones_dt = modelo_dt.predict(X_test)

# Evaluar el rendimiento del modelo
mse_dt = mean_squared_error(y_test, predicciones_dt)
r2_dt = r2_score(y_test, predicciones_dt)
print(f'Modelo Decision Tree - MSE: {mse_dt:.2f}, R2: {r2_dt:.2f}')

# Entrenar el modelo con XGBoost
modelo_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
modelo_xgb.fit(X_train, y_train)
predicciones_xgb = modelo_xgb.predict(X_test)

# Evaluar el rendimiento del modelo
mse_xgb = mean_squared_error(y_test, predicciones_xgb)
r2_xgb = r2_score(y_test, predicciones_xgb)
print(f'Modelo XGBoost - MSE: {mse_xgb:.2f}, R2: {r2_xgb:.2f}')

# Entrenar el modelo con LightGBM
modelo_lgb = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
modelo_lgb.fit(X_train, y_train)
predicciones_lgb = modelo_lgb.predict(X_test)

# Evaluar el rendimiento del modelo
mse_lgb = mean_squared_error(y_test, predicciones_lgb)
r2_lgb = r2_score(y_test, predicciones_lgb)
print(f'Modelo LightGBM - MSE: {mse_lgb:.2f}, R2: {r2_lgb:.2f}')

# Comparar los resultados
def comparar_modelos():
    print("\nResumen de la Comparación de Modelos:")
    print(f"Linear Regression - MSE: {mse_lr:.2f}, R2: {r2_lr:.2f}")
    print(f"Decision Tree - MSE: {mse_dt:.2f}, R2: {r2_dt:.2f}")
    print(f"XGBoost - MSE: {mse_xgb:.2f}, R2: {r2_xgb:.2f}")
    print(f"LightGBM - MSE: {mse_lgb:.2f}, R2: {r2_lgb:.2f}")

comparar_modelos()
