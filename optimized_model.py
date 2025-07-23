import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Cargar dataset
try:
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "karkavelrajaj/amazon-sales-dataset", "amazon.csv")
except Exception as e:
    raise SystemExit(f"No se pudo cargar el dataset: {e}")

# Limpieza de columnas numericas
for col in ["discounted_price", "actual_price", "rating_count"]:
    df[col] = df[col].replace({"[₹,]": ""}, regex=True)
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["discount_percentage"] = df["discount_percentage"].str.replace("%", "", regex=False).astype(float)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Eliminar filas con NaN en columnas clave
df.dropna(subset=["discounted_price", "actual_price", "rating", "rating_count"], inplace=True)

# Extraer la categoria principal
df["main_category"] = df["category"].str.split("|").str[0]

# Preparar caracteristicas y target
X = df[["discounted_price", "discount_percentage", "rating", "main_category"]]
y = df["rating_count"]
X = pd.get_dummies(X, columns=["main_category"], drop_first=True)

# Division en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parametros para busqueda aleatoria
def hyperparameter_search(model, param_dist, X, y):
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50,
                                cv=3, scoring="neg_root_mean_squared_error",
                                random_state=42, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_

xgb_params = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 1],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1.0, 0.1]
}

xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
best_xgb, best_params = hyperparameter_search(xgb_model, xgb_params, X_train, y_train)

# Entrenamiento final
best_xgb.fit(X_train, y_train)

# Evaluacion
y_pred = best_xgb.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("Mejores hiperparametros:", best_params)
print(f"XGBoost -> MAE: {mae:.4f} RMSE: {rmse:.4f} R2: {r2:.4f}")

# LightGBM de referencia
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = mean_squared_error(y_test, y_pred_lgb) ** 0.5
r2_lgb = r2_score(y_test, y_pred_lgb)
print(f"LightGBM -> MAE: {mae_lgb:.4f} RMSE: {rmse_lgb:.4f} R2: {r2_lgb:.4f}")
