import os
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense

# Cargar el dataset
parser = argparse.ArgumentParser(description="Comparación de modelos de regresión")
parser.add_argument(
    "--csv-path",
    default=os.getenv("AMAZON_CSV_PATH"),
    help="Ruta al archivo CSV local. Si no se proporciona se intentará descargar de Kaggle.",
)
args = parser.parse_args()

if args.csv_path:
    df = pd.read_csv(args.csv_path)
else:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "karkavelraj/amazon-sales-dataset",
            "amazon.csv",
        )
    except Exception as e:
        raise RuntimeError(
            "No se pudo cargar el dataset desde Kaggle. Use --csv-path o la variable AMAZON_CSV_PATH para especificar un archivo local. Error original: %s"
            % e
        )

# Limpieza y transformación de columnas numéricas
num_cols = {
    'discounted_price_num': ('discounted_price', '[₹,]'),
    'actual_price_num': ('actual_price', '[₹,]'),
    'discount_percentage_num': ('discount_percentage', '%'),
    'rating_num': ('rating', None),
    'rating_count_num': ('rating_count', ',')
}
for new_col, (col, repl) in num_cols.items():
    if repl:
        df[new_col] = df[col].replace({repl: ''}, regex=True).astype(float)
    else:
        df[new_col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con valores faltantes en las columnas clave
key_cols = ['rating_num', 'rating_count_num']
df.dropna(subset=key_cols, inplace=True)

# Extraer la categoría principal
df['main_category'] = df['category'].str.split('|').str[0]

# Preparar las características y la variable objetivo
X = df[['discounted_price_num', 'discount_percentage_num', 'rating_num', 'main_category']].copy()
X = pd.get_dummies(X, columns=['main_category'], drop_first=True)
y = df['rating_count_num']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo de red neuronal
model_nn = Sequential()
model_nn.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model_nn.add(Dense(units=32, activation='relu'))
model_nn.add(Dense(units=1, activation='linear'))
model_nn.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento de la red neuronal
model_nn.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo
model_nn.save('model_nn.h5')

# Predicciones con la red neuronal
y_pred_nn = model_nn.predict(X_test)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)
print("Red Neuronal -> MAE:", mae_nn, "RMSE:", rmse_nn, "R²:", r2_nn)

# Modelo XGBoost
model_xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

# Modelo LightGBM
model_lgb = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100)
model_lgb.fit(X_train, y_train)
y_pred_lgb = model_lgb.predict(X_test)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
r2_lgb = r2_score(y_test, y_pred_lgb)

# Comparación de modelos
print("\nComparación de Modelos:")
print(f"Red Neuronal vs XGBoost: \nMAE: {mae_nn} vs {mae_xgb} \nRMSE: {rmse_nn} vs {rmse_xgb} \nR²: {r2_nn} vs {r2_xgb}")
print(f"Red Neuronal vs LightGBM: \nMAE: {mae_nn} vs {mae_lgb} \nRMSE: {rmse_nn} vs {rmse_lgb} \nR²: {r2_nn} vs {r2_lgb}")
